import torch



class BYOLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, z):
        sim = (torch.nn.functional.normalize(z)*torch.nn.functional.normalize(q)).sum(1)

        return (2. - 2.*sim).mean()*2.  # *2. at the end to make (L + \tilde{L})/2. -> L + \tilde{L}
