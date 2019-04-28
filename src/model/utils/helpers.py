"""
helper functions
"""
import pdb
import pickle as pkl
import torch
from torch.autograd import Variable
from tqdm import tqdm

from . import constants as const


def prepare_vars(cuda, device, *args):
    new_args = []
    for arg in args:
        arg = Variable(arg)
        if cuda and torch.cuda.is_available():
            arg = arg.cuda()
        arg.to(device)
        new_args.append(arg)

    return new_args[0] if len(new_args) == 1 else new_args


def create_generated_data_file(generator, data_iter, outpath, cuda, device):
    """
    Generates `len(data_iter)` batches of size BATCH_SIZE from the generator. Stores the data in `output_file`
    """
    print('Creating generated data file ...')
    samples = []
    for data in tqdm(data_iter, desc=" - Create Generated Data File", leave=False):
        # sample_batch = generator.module.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN).cpu().data.numpy().tolist()
        cr_seqs, ct_seqs = data[const.CR_SEQS_KEY], data[const.CT_SEQS_KEY]
        cr_seqs, ct_seqs = prepare_vars(cuda, device, cr_seqs, ct_seqs)

        # can replace these constants with size measurements from the data batch, more safe
        tick_seqs = generator.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN, cr_seqs, ct_seqs)
        tick_seqs = tick_seqs.cpu().data.numpy().tolist()

        samples.extend(list(zip(tick_seqs, list(cr_seqs.cpu()), (ct_seqs.cpu()))))

    with open(outpath, 'wb') as fout:
        pkl.dump(samples, fout)

    return


def create_real_data_file(data_iter, outpath):
    """
    Iterates through `data_iter` and stores all its targets in `output_file`.
    """
    print('Creating real data file ...')
    samples = []
    for data in tqdm(data_iter, desc=' - Create Real Data File', leave=False):
        tick_seqs = list(data[const.SEQS_KEY].numpy())
        cr_seqs = list(data[const.CR_SEQS_KEY].numpy())
        ct_seqs = list(data[const.CT_SEQS_KEY].numpy())

        samples.extend(list(zip(tick_seqs, cr_seqs, ct_seqs)))

    with open(outpath, 'wb') as fout:
        pkl.dump(samples, fout)

    return
