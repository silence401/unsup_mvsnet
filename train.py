import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist
from config import args, device
#import cv2

#cudnn.benchmark = True
#torch.backends.cudnn.enabled = False
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
print(is_distributed)
def adjust_parameters(epoch):
    if epoch == 2-1:
        args.w_aug = 2 * args.w_aug
    elif epoch == 4-1:
        args.w_aug = 2 * args.w_aug
    elif epoch == 6-1:
        args.w_aug = 2 * args.w_aug
    elif epoch == 8-1:
        args.w_aug = 2 * args.w_aug
# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=- 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx
        #print(global_step)
        #lr_scheduler.step()
        adjust_parameters(epoch_idx)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs, depth_est = train_sample(model, model_loss, optimizer, sample, args)
            loss_photo = scalar_outputs["loss_photo"]
            loss_smooth = scalar_outputs["loss_smooth"]
            loss_ssim = scalar_outputs["loss_ssim"]
            loss_plane = scalar_outputs["loss_plane"]
            #depth_est = image_outputs['depth_est']
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    #print("dos")
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
            #loss, scalar_outputs, image_outputs, depth_est = train_sample(model, model_loss, optimizer, sample, args)
            augment_loss, scalar_outputs, image_outputs = train_sample_aug(sample, depth_est, do_summary)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary or global_step == 1:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary or global_step == 1:
                    print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, loss_photo = {:.3f}, loss_ssim = {:.3f}, loss_smooth = {:.3f}, augment_loss = {:.3f}, loss_plane = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss, loss_photo, loss_ssim, loss_smooth,
                           augment_loss,loss_plane,
                           time.time() - start_time))

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, photo_loss = {:.3f}, ssim_loss = {:.3f}, smooth_loss = {:.3f}, loss_plane = {:.3f}, time = {:3f}".format(
                                                                            epoch_idx, args.epochs,
                                                                            batch_idx,
                                                                            len(TestImgLoader), loss,
                                                                            scalar_outputs['loss_photo'], scalar_outputs['loss_ssim'], scalar_outputs['loss_smooth'],scalar_outputs["loss_plane"],
                                                                           # scalar_outputs["depth_loss"],
                                                                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()
    #print("==========proj_matrices======")
    #print(sample['proj_matrices']['stage1'].shape)
    #print("sampleimg_stage", sample['imgs_stage'][0]['stage1'].shape)
    sample_cuda = tocuda(sample)
   # ref_img = sample_cuda['imgs_stage'][0]['stage1']
   # print(ref_img.shape)
    #cv2.imwrite('ref_img.png', ref_img.cpu().detach().numpy().squeeze(0)*255)
    
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    #imgs = sample_cuda['imgs']
    #print(smaple)
    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    #print("depth_est", depth_est.shape)
    #print("====depth_est.shape========")
   # print(depth_est.shape)
    #features = outputs['features']
    #loss_dep
    loss, loss_photo, loss_ssim, loss_smooth, loss_plane = model_loss(sample_cuda["imgs"], sample_cuda["proj_matrices"], outputs)
    #print("loss:",loss)
    #loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            #print("s")
            scaled_loss.backward()
    else:
        #print("a")
        loss.backward()
     
    optimizer.step()
    scalar_outputs = {"loss": loss,
                      "loss_photo": loss_photo,
                      "loss_ssim": loss_ssim,
                      "loss_smooth": loss_smooth,
                      "loss_plane": loss_plane,
                      #"depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs), depth_est.clone().detach()

def train_sample_aug(sample, depth_est, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    # depth_gt = depth_gt.squeeze(-1)
    mask_ms = sample_cuda["mask"]
    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    mask = mask_ms["stage{}".format(num_stage)]
    # depth_interval = sample_cuda["cams"][:, 0, 1, 3, 1]

    # batch_size = sample_cuda["imgs"].size(0)
    imgs_aug = sample_cuda["imgs_aug"]
    ref_img = imgs_aug[:, 0]

    # step 1 正常训练
    # outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    # depth_est = outputs["depth"]
    # # 重建损失
    # standard_loss = criterion(sample_cuda["imgs"], sample_cuda["cams"], depth_est)
    # loss = standard_loss
    # print('standard_loss: {}'.format(standard_loss))
    # standard_loss.backward(retain_graph=True)

    # step 2 数据增强一致性训练
    # 数据增强(光照调整/gamma校正，随机噪声，随机mask)
    scale_factor = np.random.randint(3, 6) 
    ref_img, filter_mask = random_image_mask(ref_img, filter_size=(ref_img.size(2)//scale_factor, ref_img.size(3)//scale_factor))
    #print("ref_img:", ref_img)
    imgs_aug[:, 0] = ref_img
#     print(ref_img.shape)
#     from matplotlib import pyplot as plt
#     plt.imsave("augimg.png", ref_img.cpu().detach().squeeze(0).permute(1,2,0))
#     plt.imsave("augimg2.png", imgs_aug[:,1].cpu().detach().squeeze(0).permute(1,2,0))
    
    outputs_aug = model(imgs_aug, sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est_aug = outputs_aug["depth"]
    #print("depth_est_aug.shape", depth_est_aug.shape)
    # 数据增强损失
    #filter_mask = F.interpolate(filter_mask, scale_factor=0.25)
    filter_mask = filter_mask[:, 0, :, :]
    #print("filter_mask", filter_mask)
    # print('depth_est_aug: {} depth_est: {}'.format(depth_est_aug.shape, depth_est.shape))
    #loss, loss_photo, loss_ssim, loss_smooth = model_loss(sample_cuda["imgs"], sample_cuda["proj_matrices"], outputs_aug)
    augment_loss = Aug_loss(depth_est_aug, depth_est, filter_mask)
    #print('augment_loss: {}'.format(augment_loss))
    augment_loss = augment_loss * args.w_aug
    
    if is_distributed and args.using_apex:
        with amp.scale_loss(augment_loss, optimizer) as scaled_loss:
            #print("s")
            scaled_loss.backward()
    else:
        #print("a")
        augment_loss.backward()
     
    #augment_loss.backward()
    #print("loss:", loss)
    #loss.backward()

    optimizer.step()

    scalar_outputs = {"augment_loss": augment_loss}
    # print('depth_est: {}'.format(depth_est.shape))
    # print('depth_gt: {}'.format(sample["depth"].shape))
    # print('ref_img: {}'.format(sample["imgs"][:, 0].shape))
    # print('mask: {}'.format(sample["mask"].shape))
    #print(type(depth_est_aug))
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)
    image_outputs = {"depth_est_aug": depth_est_aug * mask * filter_mask}

    return tensor2float(augment_loss), tensor2float(scalar_outputs), image_outputs

@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    #loss = model_loss(sample_cuda["imgs"], sample_cuda["features"], sample_cuda["proj_matrices"], depth_est)
    loss, loss_photo, loss_ssim, loss_smooth, loss_plane = model_loss(sample_cuda["imgs"], sample_cuda["proj_matrices"], outputs)
    #loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    scalar_outputs = {"loss": loss,
                      "loss_photo": loss_photo,
                      "loss_ssim": loss_ssim,
                      "loss_smooth": loss_smooth,
                      "loss_plane": loss_plane,
                     # "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
                      "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    #args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    #device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method)
    model.to(device)
    #model_loss = cas_mvsnet_loss
    model_loss = UnSupLoss(args, dlossw=[float(e) for e in args.dlossw.split(",") if e])
    model_loss.to(device)
    Aug_loss = AugLoss()
    Aug_loss.to(device)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, args.numdepth, args.interval_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError