from tqdm import tqdm
import paddle
from x2paddle.torch2paddle import DataLoader
from core.option import parser
from core.model import WDSR_A
# from core.model import WDSR_B
from core.data.div2k import DIV2K
from core.data.utils import quantize
from core.utils import AverageMeter
from core.utils import calc_psnr
from core.utils import load_checkpoint
from core.utils import load_weights


def forward(x):
    with paddle.no_grad():
        sr = model(x)
        return sr


def forward_x8(x):
    x = x.squeeze(0).permute(1, 2, 0)
    with paddle.no_grad():
        sr = []
        for rot in range(0, 4):
            for flip in [False, True]:
                _x = x.flip([1]) if flip else x
                _x = _x.rot90(rot)
                out = model(_x.permute(2, 0, 1).unsqueeze(0)).squeeze(0
                    ).permute(1, 2, 0)
                out = out.rot90(4 - rot)
                out = out.flip([1]) if flip else out
                sr.append(out)
        return paddle.stack(sr).mean(0).permute(2, 0, 1)
        

def test(dataset, loader, model, args,  tag=''):
    psnr = AverageMeter()
    model.eval()
    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)
        for data in loader:
            lr, hr = data
            lr = lr
            hr = hr
            # if args.self_ensemble:
            #     sr = forward_x8(lr)
            # else:
            #     sr = forward(lr)
            sr = forward(lr)
            sr = quantize(sr, args.rgb_range)
            psnr.update(calc_psnr(sr, hr, scale=args.scale, max_value=args.
                rgb_range[1]), lr.shape[0])
            t.update(lr.shape[0])
    print('DIV2K (val) PSNR: {:.4f} dB'.format(psnr.avg.numpy()[0]))


if __name__ == '__main__':
    parser.add_argument('--dataset-dir', type=str, required=True, help='DIV2K Dataset Root Directory')
    parser.add_argument('--checkpoint-file', type=str, required=True)
    parser.add_argument('--self-ensemble', action='store_true')
    args = parser.parse_args()
    # device=paddle.CUDAPlace(0)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if args.model == 'WDSR-B':
    #     model = WDSR_B(args)
    # else:
    #     model = WDSR_A(args)
    model = WDSR_A(args)
    model = load_weights(model, load_checkpoint(args.checkpoint_file)['state_dict'])
    dataset = DIV2K(args, train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    test(dataset, dataloader, model, args)
