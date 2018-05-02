from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data import TabularDataset

DATA_ROOT = "./datasets/"


def load_dataset(db_name, batch_size):
    """
    Load the csv datasets into torchtext files

    Inputs:
    db_name (string)
       The name of the dataset. This name must correspond to the folder name.
    batch_size
       The batch size
    """
    print "Loading " + db_name + "..."

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    tv_datafields = [("sentence", TEXT),
                     ("label", LABEL)]

    trn, vld = TabularDataset.splits(
        path=DATA_ROOT + db_name,  # the root directory where the data lies
        train='train.csv', validation="test.csv",
        format='csv',
        skip_header=False,
        fields=tv_datafields)

    TEXT.build_vocab(trn)

    print "vocab size: %i" % len(TEXT.vocab)

    train_iter, val_iter = BucketIterator.splits(
        (trn, vld), 
        batch_sizes=(batch_size, batch_size),
        device=-1,  # specify dont use gpu
        sort_key=lambda x: len(x.sentence),  # sort the sentences by length
        sort_within_batch=False,
        repeat=False)

    return train_iter, val_iter
