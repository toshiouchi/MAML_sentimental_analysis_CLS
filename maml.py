import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    '''
    自己アテンション
    dim_hidden: 入力特徴量の次元
    num_heads : マルチヘッドアテンションのヘッド数
    qkv_bias  : クエリなどを生成する全結合層のバイアスの有無
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 qkv_bias: bool=False):
        super().__init__()

        # 特徴量を各ヘッドのために分割するので、
        # 特徴量次元をヘッド数で割り切れるか検証
        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads

        # ヘッド毎の特徴量次元
        dim_head = dim_hidden // num_heads

        # ソフトマックスのスケール値
        self.scale = dim_head ** -0.5

        # ヘッド毎にクエリ、キーおよびバリューを生成するための全結合層
        self.proj_in = nn.Linear(
            dim_hidden, dim_hidden * 3, bias=qkv_bias)

        # 各ヘッドから得られた特徴量を一つにまとめる全結合層
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor, attention_ids:torch.Tensor):
        bs, ns = x.shape[:2]

        qkv = self.proj_in(x)

        # view関数により
        # [バッチサイズ, 特徴量数, QKV, ヘッド数, ヘッドの特徴量次元]
        # permute関数により
        # [QKV, バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元]
        qkv = qkv.view(
            bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # クエリ、キーおよびバリューに分解
        q, k, v = qkv.unbind(0)

        # クエリとキーの行列積とアテンションの計算(今回マスクは不使用)
        # attnは[バッチサイズ, ヘッド数, 特徴量数, 特徴量数]
        attn = q.matmul(k.transpose(-2, -1))
        attn = attn + torch.unsqueeze( torch.unsqueeze( attention_ids * -1e9 , dim = 1 ), dim = 1 )
        attn = (attn * self.scale).softmax(dim=-1)

        # アテンションとバリューの行列積によりバリューを収集
        # xは[バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元]
        x = attn.matmul(v)

        # permute関数により
        # [バッチサイズ, 特徴量数, ヘッド数, ヘッドの特徴量次元]
        # flatten関数により全てのヘッドから得られる特徴量を連結して、
        # [バッチサイズ, 特徴量数, ヘッド数 * ヘッドの特徴量次元]
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj_out(x)

        return x


class FNN(nn.Module):
    '''
    Transformerエンコーダ内の順伝播型ニューラルネットワーク
    dim_hidden     : 入力特徴量の次元
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden: int, dim_feedforward: int):
        super().__init__()

        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class TransformerEncoderLayer(nn.Module):
    '''
    Transformerエンコーダ層
    dim_hidden     : 入力特徴量の次元
    num_heads      : ヘッド数
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 dim_feedforward: int):
        super().__init__()

        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor, attention_ids: torch.Tensor):
        x = self.norm1(x)
        x = self.attention(x, attention_ids) + x
        x = self.norm2(x)
        x = self.fnn(x) + x
        
        return x 

        return x

class PositionalEncoding(nn.Module):
    '''
    位置エンコーディング （Positional encoding）
    dim_embedding: 埋込み次元
    max_len      : 入力の最大系列長
    temperature  : 温度定数
    '''
    def __init__(self, dim_embedding: int,
                 max_len: int=5000, temperature=10000):
        super().__init__()

        assert dim_embedding % 2 == 0

        dim_t = torch.arange(0, dim_embedding, 2)
        dim_t = dim_t / dim_embedding
        dim_t = temperature ** dim_t

        x_encoding = torch.arange(max_len).unsqueeze(1)
        x_encoding = x_encoding / dim_t

        # 位置情報を保持するテンソル

        pe = torch.zeros(max_len, dim_embedding)
        pe[:, ::2] = x_encoding.sin()
        pe[:, 1::2] = x_encoding.cos()

        # PEをメモリに保存
        self.register_buffer('pe', pe)

    '''
    位置エンコーディングの順伝播
    x: 位置エンコーディングを埋め込む対象のテンソル,
       [バッチサイズ, 系列長, 埋め込み次元]
    '''
    def forward(self, x: torch.Tensor):
        seq = x.shape[1]
        x = x + self.pe[:seq]

        return x    
    

class fSelfAttention(nn.Module):
    '''
    自己アテンション
    dim_hidden: 入力特徴量の次元
    num_heads : マルチヘッドアテンションのヘッド数
    qkv_bias  : クエリなどを生成する全結合層のバイアスの有無
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 qkv_bias: bool=False):
        super().__init__()

        # 特徴量を各ヘッドのために分割するので、
        # 特徴量次元をヘッド数で割り切れるか検証
        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads

        # ヘッド毎の特徴量次元
        dim_head = dim_hidden // num_heads

        # ソフトマックスのスケール値
        self.scale = dim_head ** -0.5

        # ヘッド毎にクエリ、キーおよびバリューを生成するための全結合層
        #self.proj_in = nn.Linear(
        #    dim_hidden, dim_hidden * 3, bias=qkv_bias)

        # 各ヘッドから得られた特徴量を一つにまとめる全結合層
        #self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor, attention_ids: torch.Tensor, i, weights ):
        bs, ns = x.shape[:2]

        #qkv = self.proj_in(x)
        qkv = F.linear(x, weights['layers.' + str(i) + '.attention.proj_in.weight'], bias = None)


        # view関数により
        # [バッチサイズ, 特徴量数, QKV, ヘッド数, ヘッドの特徴量次元]
        # permute関数により
        # [QKV, バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元]
        qkv = qkv.view(
            bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # クエリ、キーおよびバリューに分解
        q, k, v = qkv.unbind(0)

        # クエリとキーの行列積とアテンションの計算(今回マスクは不使用)
        # attnは[バッチサイズ, ヘッド数, 特徴量数, 特徴量数]
        attn = q.matmul(k.transpose(-2, -1))
        attn = attn + torch.unsqueeze( torch.unsqueeze( attention_ids * -1e9 , dim = 1 ), dim = 1 )
        attn = (attn * self.scale).softmax(dim=-1)

        # アテンションとバリューの行列積によりバリューを収集
        # xは[バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元]
        x = attn.matmul(v)

        # permute関数により
        # [バッチサイズ, 特徴量数, ヘッド数, ヘッドの特徴量次元]
        # flatten関数により全てのヘッドから得られる特徴量を連結して、
        # [バッチサイズ, 特徴量数, ヘッド数 * ヘッドの特徴量次元]
        x = x.permute(0, 2, 1, 3).flatten(2)
        #x = self.proj_out(x)
        x = F.linear(x, weights['layers.' + str(i) + '.attention.proj_out.weight'], weights['layers.' + str(i) + '.attention.proj_out.bias'])

        return x

class fFNN(nn.Module):
    '''
    Transformerエンコーダ内の順伝播型ニューラルネットワーク
    dim_hidden     : 入力特徴量の次元
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden: int, dim_feedforward: int):
        super().__init__()

        #self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        #self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor, i, weights ):
        #x = self.linear1(x)
        x = F.linear(x, weights['layers.' + str(i) + '.fnn.linear1.weight'], weights['layers.' + str(i) + '.fnn.linear1.bias'])
        x = self.activation(x)
        #x = self.linear2(x)
        x = F.linear(x, weights['layers.' + str(i) + '.fnn.linear2.weight'], weights['layers.' + str(i) + '.fnn.linear2.bias'])

        return x

class fTransformerEncoderLayer(nn.Module):
    '''
    Transformerエンコーダ層
    dim_hidden     : 入力特徴量の次元
    num_heads      : ヘッド数
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 dim_feedforward: int):
        super().__init__()

        self.fattention = fSelfAttention(dim_hidden, num_heads)
        self.ffnn = fFNN(dim_hidden, dim_feedforward)
        self.dim_hidden = dim_hidden

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor, attention_ids: torch.Tensor, i, weights ):
        #x = self.norm1(x, 1, weights)
        x = F.layer_norm(x, (self.dim_hidden,), weight=weights['layers.' + str(i) + '.norm1.weight'], bias=weights['layers.' + str(i) + '.norm1.bias'], eps=1e-05)
        x = self.fattention(x, attention_ids, i, weights ) + x
        #x = self.norm2(x)
        x = F.layer_norm(x, (self.dim_hidden,), weight=weights['layers.' + str(i) + '.norm2.weight'], bias=weights['layers.' + str(i) + '.norm2.bias'], eps=1e-05)
        x = self.ffnn(x, i, weights) + x

        return x

class MAML(nn.Module):
        
    def __init__(self):
        super(MAML, self).__init__()
        num_class = 2
        num_heads = 4
        num_layers = 6
        self.num_layers = num_layers
        dim_hidden = 256
        self.dim_hidden = dim_hidden
        dim_feedforward = 384
        max_seq =128
        self.max_seq = max_seq
        self.embed = nn.Embedding( 30522, dim_hidden )
        self.pe = PositionalEncoding( dim_hidden )
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            dim_hidden, num_heads, dim_feedforward
        ) for _ in range(num_layers)])
        self.ftrenc = fTransformerEncoderLayer( dim_hidden, num_heads, dim_feedforward )

        self.logits = nn.Linear( dim_hidden * max_seq, num_class )
        #self._reset_parameters()

    #def _reset_parameters(self):
    #    print( "execute reset parameters:" )
    #    for module in self.modules():
    #        if isinstance(module, nn.Linear):
    #            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #            if module.bias is not None:
    #                nn.init.zeros_(module.bias)
    #            #nn.init.xavier_normal_(module.weight)
    #            #nn.init.xavier_normal_(module.bias)
    #        elif isinstance(module, nn.LayerNorm):
    #            nn.init.zeros_(module.bias)
    #            nn.init.ones_(module.weight)
   

    def forward(self, x, attention_ids):
        
        x = self.embed( x )
        x = self.pe( x )

        # Transformerエンコーダ層を適用
        for layer in self.layers:
            x = layer(x, attention_ids)
            #print( "layer x:", x )
      
        x = x.view( x.size(0), -1 )
     
        return self.logits(x)

    def adaptation(self, x, attention_ids, weights):
        x = F.embedding(x, weights['embed.weight'] )
        x = self.pe( x )
        for block in range( self.num_layers ):
            x = self.ftrenc(x, attention_ids, block, weights )
        
        x = x.view( x.size(0), -1 )
        
        return F.linear(x, weights['logits.weight'], weights['logits.bias'])
