import torch
from transformers.models.dab_detr.modeling_dab_detr import inverse_sigmoid

def embed_circle(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    angles = input_ids * 2 * torch.pi / q
    cos_component = torch.cos(angles)
    sin_component = torch.sin(angles)
    circle = torch.stack((cos_component, sin_component), dim=-1)
    R = torch.randn(2,d_model)
    embedding_table[input_ids] = torch.matmul(circle,R)
    return embedding_table

def embed_inverse_add(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    base = torch.randn((q + 1) // 2, d_model)
    inverse = -torch.flip(base[1:], dims=[0])
    embedding_table[input_ids] = torch.cat((base, inverse), dim=0)
    return embedding_table

def embed_inverse_mul(q: int, d_model: int) -> torch.Tensor:
    """乗法逆元でペアを作り、互いに反対向きのベクトルを割り当てる。自己逆元はそのまま。"""
    embedding_table = torch.zeros(q, d_model)
    filled = torch.zeros(q, dtype=torch.bool)
    for i in range(1, q):  # 0 は未使用
        if filled[i]:
            continue
        inv = pow(i, -1, q)
        vec = torch.randn(1, d_model)
        embedding_table[i] = vec
        filled[i] = True
        if inv != i:  # 非自己逆元は反対向きで対になる
            embedding_table[inv] = -vec
            filled[inv] = True
    return embedding_table

def embed_square(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    base = torch.randn((q + 1) // 2, d_model)
    inverse = torch.flip(base[1:], dims=[0])
    embedding_table[input_ids] = torch.cat((base, inverse), dim=0)
    return embedding_table

# 逆元を反対にすることに意味はあるか  なさそう
def embed_fold_half_noflip(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    base = torch.randn((q + 1) // 2, d_model)
    inverse = -base[1:]
    embedding_table[input_ids] = torch.cat((base, inverse), dim=0)
    return embedding_table

def embed_fold_half_random(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    base = torch.randn((q + 1) // 2, d_model)
    perm = torch.randperm((q - 1)//2)
    inverse = base[1:]
    inverse = -inverse[perm]
    embedding_table[input_ids] = torch.cat((base, inverse), dim=0)
    return embedding_table

def embed_fold_half_allrandom(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    perm = torch.randperm(q)
    base = torch.randn((q + 1) // 2, d_model)
    inverse = -base[1:]
    embedding_table[perm] = torch.cat((base, inverse), dim=0)
    return embedding_table

def rotate_quarter(q: int, d_model: int) -> torch.Tensor:
    rotate = torch.tensor([[0., 1.], [-1., 0.]])
    if (q - 1) % 4 == 0:
        base = torch.randn((q - 1) // 4 + 1, d_model)
        rotate_base = base[1:].reshape(-1, d_model//2, 2)
        rotate90 = torch.matmul(rotate_base, rotate.T)
        rotate180 = torch.matmul(rotate90, rotate.T)
        rotate270 = torch.matmul(rotate180, rotate.T)
        return base, rotate90.reshape(-1, d_model), rotate180.reshape(-1, d_model), rotate270.reshape(-1, d_model)
    else:
        base = torch.randn((q + 1) // 4 + 1, d_model)    
        rotate_base = base[1:].reshape(-1, d_model//2, 2)
        rotate90 = torch.matmul(rotate_base, rotate.T)
        rotate180 = torch.matmul(rotate90, rotate.T)
        rotate270 = torch.matmul(rotate180, rotate.T)
        return base, rotate90.reshape(-1, d_model)[:-1,:], rotate180.reshape(-1, d_model), rotate270.reshape(-1, d_model)[:-1,:]
    
   

# 情報を折りたたむ方法に意味はあるか, 1/4にする方法
def fold_quarter(q: int, d_model: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (q - 1) % 4 == 0:
        quadrant1 = torch.randn((q - 1) // 4 + 1, d_model)
        quadrant2 = torch.cat((-quadrant1[1:,:d_model//2], quadrant1[1:,d_model//2:]), dim=1)
        quadrant3 = -quadrant1[1:]
        quadrant4 = torch.cat((-quadrant3[:,:d_model//2], quadrant3[:,d_model//2:]), dim=1)
    else:
        quadrant1 = torch.randn((q + 1) // 4 + 1, d_model)
        quadrant2 = torch.cat((-quadrant1[1:-1,:d_model//2], quadrant1[1:-1,d_model//2:]), dim=1)
        quadrant3 = -quadrant1[1:]
        quadrant4 = torch.cat((-quadrant3[:-1,:d_model//2], quadrant3[:-1,d_model//2:]), dim=1)
    
    return quadrant1, quadrant2, quadrant3, quadrant4

def embed_fold_quarter_noflip(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    quadrant1, quadrant2, quadrant3, quadrant4 = fold_quarter(q, d_model)
    embedding_table[input_ids] = torch.cat((quadrant1, quadrant2, quadrant3, quadrant4), dim=0)
    return embedding_table

def embed_fold_quarter_origami_flip(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    quadrant1, quadrant2, quadrant3, quadrant4 = fold_quarter(q, d_model)
    embedding_table[input_ids] = torch.cat((quadrant1, quadrant2, torch.flip(quadrant3, dims=[0]), torch.flip(quadrant4, dims=[0])), dim=0)
    return embedding_table

def embed_fold_quarter_allrandom(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    quadrant1, quadrant2, quadrant3, quadrant4 = fold_quarter(q, d_model)
    perm = torch.randperm(q)
    embedding_table[perm] = torch.cat((quadrant1, quadrant2, quadrant3, quadrant4), dim=0)
    return embedding_table

def embed_rotate_quarter_noflip(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    base, rotate90, rotate180, rotate270 = rotate_quarter(q, d_model)
    embedding_table[input_ids] = torch.cat((base, rotate90, rotate180, rotate270), dim=0)
    return embedding_table

def embed_rotate_quarter_allrandom(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    base, rotate90, rotate180, rotate270 = rotate_quarter(q, d_model)
    perm = torch.randperm(q)
    embedding_table[perm] = torch.cat((base, rotate90, rotate180, rotate270), dim=0)
    return embedding_table

def embed_rotate_quarter_origami_flip(q: int, d_model: int) -> torch.Tensor:
    embedding_table = torch.zeros(q, d_model)
    input_ids = torch.arange(q)
    base, rotate90, rotate180, rotate270 = rotate_quarter(q, d_model)
    embedding_table[input_ids] = torch.cat((base, rotate90, torch.flip(rotate180, dims=[0]), torch.flip(rotate270, dims=[0])), dim=0)
    return embedding_table

def generate_embedding_table(
    q: int,
    d_model: int,
    embed_type: str = "inverse_mul",
    scale: float = 0.02,
    seed: int | None = None,
) -> torch.Tensor:
    """embed_type に応じて埋め込み表を生成し、標準偏差を scale 倍に調整する。"""
    if seed is not None:
        torch.manual_seed(seed)

    embed_type = embed_type.lower()
    if embed_type == "circle":
        embedding_table = embed_circle(q, d_model)
    elif embed_type == "inverse_add":
        embedding_table = embed_inverse_add(q, d_model)
    elif embed_type == "inverse_mul":
        embedding_table = embed_inverse_mul(q, d_model)
    elif embed_type in ("square", "inverse_square"):
        embedding_table = embed_square(q, d_model)
    else:
        raise ValueError(
            f"未知の embed_type です: {embed_type}. "
            "circle / inverse_add / inverse_mul / square から選んでください。"
        )

    if scale is not None:
        embedding_table = embedding_table * float(scale)
    return embedding_table