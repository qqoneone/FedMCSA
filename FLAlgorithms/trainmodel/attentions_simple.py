import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy


def cal_scores(query, key, hyper_c, num_of_method):
    "Compute scores between query and key"
    if num_of_method == 0:
        # method 0: scaled dot
        d_k = query.size(-1)
        scores = hyper_c * torch.matmul(query, key) / math.sqrt(d_k)
    elif num_of_method == 1:
        # method 1: cosine similarity
        # Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
        #                          = dot(u / norm(u), v / norm(v))
        # We fist normalize the rows, before computing their dot products via transposition:
        a_norm = query / query.norm(dim=1)[:, None]
        b_norm = key / key.norm(dim=1)[:, None]
        scores = hyper_c * torch.mm(a_norm, b_norm.transpose(0, 1))
    elif num_of_method == 2:
        # method 2: exponential cosine similarity function
        a_norm = query / query.norm(dim=1)[:, None]
        b_norm = key / key.norm(dim=1)[:, None]
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        # scores = torch.exp(hyper_c * cos_sim)
        scores = hyper_c * cos_sim
    elif num_of_method == 3:
        # method 3: exponential cosine similarity / exp function
        a_norm = query / query.norm(dim=1)[:, None]
        b_norm = key / key.norm(dim=1)[:, None]
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        # hyper_c = torch.tensor(hyper_c, dtype=torch.float)
        # scores = torch.exp(hyper_c * cos_sim) / torch.exp(hyper_c)
        scores = hyper_c * (cos_sim - 1)
    else:
        # method 2: pearson correlation coefficient
        temp = 0

    return scores


def attention_simple(query, key, value, hyper_c, num_of_method, mask=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = cal_scores(query, key, hyper_c, num_of_method)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    res_query = torch.matmul(p_attn, value)
    return res_query, p_attn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attn = None

    def forward(self, query, key, value, hyper_c=1, num_of_method=2, mask=None):
        # hyper_c = torch.tensor(hyper_c, dtype=torch.int)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        x, self.attn = attention_simple(query, key, value, hyper_c, num_of_method, mask=mask)

        # x = (num_of_client, num_of_parameters_model), self.attn = (num_of_client, num_of_client)
        return x, self.attn

def call_attentions_simple(input, hyper_c=3.5, num_of_method = 2, num_of_iter = 20, flag_test = 1):
    query = copy.deepcopy(input)
    key = copy.deepcopy(query)
    value = copy.deepcopy(query)
    for i in range(num_of_iter):
        m, attn = model(query, key, value, hyper_c, num_of_method)
        if flag_test == 1:
            # for test
            error = m - query
            print("-" * 20 + str(i))
            print(torch.sum(error, dim=1))
            print(error)
            print(attn)

        query = m


if __name__ == "__main__":

    h = 8
    d_model = 2500
    batch_size = 1
    seq_length = 20
    model = Attention()

    torch.manual_seed(0)

    query = torch.randn([seq_length, d_model])

    call_attentions_simple(query)


