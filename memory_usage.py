import gc

import torch


def to_gb(size):
    return size/((1024**3) * 8)

def test(x, y):
    print('in')
    print(x)
    print(y)
    print(x*y)
    print('out')

    return x/0

def get_memory_usage():
    cuda_total = 0
    cpu_total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if len(obj.size()) > 0:
                    if obj.type() == 'torch.FloatTensor':
                        print(obj.size())
                        print(obj.type())
                        cpu_total += reduce(test, obj.size()) * 32
                    elif obj.type() == 'torch.LongTensor':
                        cpu_total += reduce(test, obj.size()) * 64
                    elif obj.type() == 'torch.IntTensor':
                        cpu_total += reduce(test, obj.size()) * 32
                    # ----------- CUDA ----------
                    if obj.type() == 'torch.cuda.FloatTensor':
                        cuda_total += reduce(lambda x, y: x*y, obj.size()) * 32
                    elif obj.type() == 'torch.cuda.LongTensor':
                        cuda_total += reduce(lambda x, y: x*y, obj.size()) * 64
                    elif obj.type() == 'torch.cuda.IntTensor':
                        cuda_total += reduce(lambda x, y: x*y, obj.size()) * 32
                    #else:
                    # Few non-cuda tensors in my case from dataloader
        except Exception as e:
            pass
    return to_gb(cpu_total), to_gb(cuda_total)