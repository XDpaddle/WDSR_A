import paddle
import shutil


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_psnr(a, b, scale=None, max_value=255.0):
    if scale:
        shave = scale + 6
        a = a[..., shave:-shave, shave:-shave]
        b = b[..., shave:-shave, shave:-shave]
    return 10.0 * (max_value ** 2 / ((a - b) ** 2).mean()).log10()


def adjust_lr(optimizer, lr, step, decay_steps, decay_gamma):
    current_lr = lr * decay_gamma ** ((step + 1) // decay_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return current_lr


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg['lr']


def save_checkpoint(state, path, is_best):
    paddle.save(state, path)
    if is_best:
        shutil.copyfile(path, path.replace('latest.pdiparams',
            'best.pdiparams'))


def load_checkpoint(path):
    return paddle.load(path)


def load_weights(model, state_dict):
    _state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in _state_dict.keys():
            _state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model
