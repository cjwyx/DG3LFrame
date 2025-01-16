import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[1]).to(support.device).unsqueeze(0), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("bnm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    
class AGCRNCell_gatembedding3(nn.Module):
    def __init__(self, node_num, dim_in,dim_state, dim_out,gate_emb_dim, cheb_k):
        super(AGCRNCell_gatembedding3, self).__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.node_num = node_num
        self.state_dim = dim_state
        self.mlpgate = nn.Linear(dim_in+self.state_dim*2+gate_emb_dim, dim_out)
        self.gate = AGCN(dim_in+self.state_dim*2, 3*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.state_dim*2, dim_out, cheb_k)

    def forward(self, x, state1,state2,gatembedding, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, state_dim
        #gatembedding: B, num_nodes, embedding_dim
        state1 = state1.to(x.device)
        state2 = state2.to(x.device)
        input_and_state_with_gateembedding = torch.cat((x, state1,state2,gatembedding), dim=-1)
        mr = torch.sigmoid(self.mlpgate(input_and_state_with_gateembedding))
        state = mr*state1 + (1-mr)*state2
        
        
        
        X = torch.cat((x, state1,state2), dim=-1)
        zz_r = torch.sigmoid(self.gate(X, supports))
        z1,z2, r = torch.split(zz_r, self.dim_out, dim=-1)
        candidate = torch.cat((x, z1*state1,z2*state2), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h


    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    

class ADCRNN_Decoder4(nn.Module):
    def __init__(self, node_num, dim_in,dim_state, dim_out,gate_emb_dim,hiddenoutput_dim, cheb_k):
        '''
        
        '''
        super(ADCRNN_Decoder4, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        # self.num_layers = num_layers
        self.dcrnn_cell = AGCRNCell_gatembedding3(node_num, dim_in,dim_state,dim_out,gate_emb_dim,cheb_k)
        self.hop_proj = nn.Linear(dim_out,hiddenoutput_dim,bias=True)

    def forward(self, xt, init_state1,init_state2,gatembedding, supports):
        # xt: (B, N, D)
        # init_state: (B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_state = self.dcrnn_cell(xt,init_state1,init_state2,gatembedding, supports)
        transnext_state = self.hop_proj(current_state)
        return current_state, transnext_state    

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
    
class TGAT(nn.Module):
    def __init__(self,d_f,n_time,n_ghead,feed_forward_dim,d_out,dropgat=0.25):
        super(TGAT, self).__init__()
        
        self.mlp1input_dim = n_time*d_f
        self.channellinear1 = nn.Linear(self.mlp1input_dim,feed_forward_dim,bias=True)
        self.act1 = nn.GELU()
        self.channellinear2 = nn.Linear(feed_forward_dim,d_out,bias=False)
        self.gat = SelfAttentionLayer(d_out,feed_forward_dim,n_ghead,dropgat)
        
        # self.ln1 = nn.LayerNorm(self.d_out)
    def forward(self,X):
        '''
        X (...,n_time,n_node,d_f) -> (...,n_node,d_out)
        '''
        batch_size,n_time,n_node,d_f = X.shape
        # (b,t,n,d) -> (b,n,t,d) -> (b,n,t*d)
        X = X.permute(0,2,1,3).reshape(batch_size,n_node,-1).contiguous()
        X = self.channellinear2(self.act1(self.channellinear1(X)))
        H = self.gat.forward(X,dim=-2)
        
        
        return H


class DGGGL(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, gatoutput=32,num_layers=1, cheb_k=3,
                 ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=3000, use_curriculum_learning=True,outputhidden=32,gathidden=64,
                steps_per_day=288,
                input_embedding_dim=8,
                tod_embedding_dim=8,
                dow_embedding_dim=8,
                spatial_embedding_dim=0,
                adaptive_embedding_dim=8,
                feed_forward_dim = 156,
                num_layers_t = 1,
                num_layers_s = 1,
                num_at_heads = 2,
                adaptive_embedding_dim_for_gate=8
                 ):
        super(DGGGL, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.gatoutput = gatoutput
        self.gathidden = gathidden
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning,
        # self.dec_layers_num = dec_layers_num
        self.outputhidden = outputhidden
        
        self.in_steps = horizon
        self.out_steps = horizon
        self.steps_per_day = steps_per_day
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.dim_formerout = rnn_units
        self.adaptive_embedding_dim_for_gate = adaptive_embedding_dim_for_gate

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            # + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, num_nodes, adaptive_embedding_dim))
            )
        if adaptive_embedding_dim_for_gate > 0:
            self.adaptive_embedding_for_gate = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, num_nodes, adaptive_embedding_dim_for_gate))
            )
        else:
            raise ValueError("adaptive_embedding_dim_for_gate must be greater than 0")

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()
        # self.gatflatoutput = gatoutput*dec_layers_num
        self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.dim_formerout
            )
        self.query_gp_proj = nn.Linear(
            self.model_dim, self.dim_formerout)
            
        # self.rx2mem_linear = nn.Linear(self.model_dim, self.mem_dim, bias=False)
        # encoder
        self.encoder = TGAT(
                        d_f= self.input_dim,
                        n_time=self.in_steps,
                        n_ghead=4,
                        feed_forward_dim=feed_forward_dim,
                        d_out=self.rnn_units,
                        dropgat=0)
        
        # deocoder
        self.decoder_dim = self.rnn_units + self.dim_formerout
        self.decoder = ADCRNN_Decoder4(
            node_num=self.num_nodes,
            dim_in=self.output_dim + self.ycov_dim,
            dim_state=self.rnn_units,
            dim_out=self.rnn_units,
            gate_emb_dim = self.adaptive_embedding_dim_for_gate, 
            hiddenoutput_dim=self.rnn_units,
            cheb_k=self.cheb_k)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_at_heads, 0)
                for _ in range(num_layers_t)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_at_heads, 0)
                for _ in range(num_layers_s)
            ]
        )

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        # output
        self.proj = nn.Sequential(
            nn.Linear(self.rnn_units, self.output_dim, bias=True),
            
        )
        # self.gcrn_emb = nn.Linear(self.output_dim,self.gcrnoutput,bias=False)
        
        # self.output_proj1 = nn.Linear(horizon*(self.gcrnoutput+self.gatoutput*self.dec_layers_num),horizon*outputhidden,bias=True)
        # self.gelu = nn.GELU()
        # self.output_proj2 = nn.Linear(horizon*outputhidden,horizon*output_dim,bias=False)
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        # memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['Wqg1'] = nn.Parameter(torch.randn(self.dim_formerout, self.mem_dim), requires_grad=True) # project memory to embedding (bs,N,mem)
        memory_dict['Wqg2'] = nn.Parameter(torch.randn(self.dim_formerout, self.mem_dim), requires_grad=True) # project memory to embedding (bs,N,mem)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
        return value, query, pos, neg
    def query_graph_and_memory(self, h_t:torch.Tensor):
        query1 = torch.matmul(h_t, self.memory['Wqg1'])     # (B, N, d)
        query2 = torch.matmul(h_t, self.memory['Wqg2'])     # (B, N, d)
        att_score1 = torch.softmax(torch.matmul(query1, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        att_score2 = torch.softmax(torch.matmul(query2, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        node_embeddings1 = torch.matmul(att_score1, self.memory['Memory'])     # (B, N, d)
        node_embeddings2 = torch.matmul(att_score2, self.memory['Memory'])     # (B, N, d)
        g1 = F.softmax(F.relu(torch.matmul(node_embeddings1, node_embeddings2.transpose(-1,-2))), dim=-1)
        g2 = F.softmax(F.relu(torch.matmul(node_embeddings2, node_embeddings1.transpose(-1,-2))), dim=-1)
        _, ind1 = torch.topk(att_score1, k=2, dim=-1)
        _, ind2 = torch.topk(att_score2, k=2, dim=-1)
        pos1 = self.memory['Memory'][ind1[:, :, 0]] # B, N, d
        neg1 = self.memory['Memory'][ind1[:, :, 1]] # B, N, d
        pos2 = self.memory['Memory'][ind2[:, :, 0]] # B, N, d
        neg2 = self.memory['Memory'][ind2[:, :, 1]] # B, N, d
        supports = [g1, g2]
        return g1,g2,query1,query2,pos1,pos2,neg1,neg2
    def query_dyngraph(self, H:torch.Tensor):
        query1 = torch.matmul(H, self.memory['Wqg1'])     # (B,T, N, d)
        query2 = torch.matmul(H, self.memory['Wqg2'])     # (B,T, N, d)
        att_score1 = torch.softmax(torch.matmul(query1, self.memory['Memory'].t()), dim=-1)         # alpha: (B,T, N, M)
        att_score2 = torch.softmax(torch.matmul(query2, self.memory['Memory'].t()), dim=-1)         # alpha: (BT, N, M)
        node_embeddings1 = torch.matmul(att_score1, self.memory['Memory'])     # (B,T, N, d)
        node_embeddings2 = torch.matmul(att_score2, self.memory['Memory'])     # (B,T, N, d)
        
        logits1 = F.relu(torch.matmul(node_embeddings1, node_embeddings2.transpose(-1,-2)))
        logits2 = F.relu(torch.matmul(node_embeddings2, node_embeddings1.transpose(-1,-2)))
        g1 = F.softmax(logits1, dim=-1) # (B,T,N,N)
        g2 = F.softmax(logits2, dim=-1) # (B,T,N,N)
        
        # permute to (T,B,N,N)
        g1 = g1.permute(1,0,2,3)
        g2 = g2.permute(1,0,2,3)
        return g1,g2,logits1.permute(1,0,2,3),logits2.permute(1,0,2,3)
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
    # def forward(self, x, y_cov, labels=None, batches_seen=None):
        x = history_data[..., [0]]
        # batch_size,T_size,Node_size,F_dim = x.shape
        batchsize,T_size,N_size = x.shape[:-1]
        y_cov = future_data[..., [1]]
        labels = future_data[..., [0]]
        rx = history_data[..., : self.input_dim]
        rx = self.input_proj(rx)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [rx]

        if self.tod_embedding_dim > 0:
            tod = history_data[..., 1]
        if self.dow_embedding_dim > 0:
            dow = history_data[..., 2]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batchsize, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batchsize, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        
        rx = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        
        for attn in self.attn_layers_t:
            rx = attn(rx, dim=1)
        for attn in self.attn_layers_s:
            rx = attn(rx, dim=2)
        rx_g = self.query_gp_proj(rx) # (batch_size,T, num_nodes, mem_dim)
        
        rx_t = rx.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        rx_t = rx_t.reshape(
            batchsize, self.num_nodes, -1
        )
        
            
        rx = self.output_proj(rx_t).view(batchsize, self.num_nodes, self.out_steps, -1)
        rx = rx.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        # rx = torch.concat([rx,tem_emb],dim=-1) #(B,T,N,output_dim+temproal_embedding_dim)
        # init_state = self.encoder.init_hidden(x.shape[0])
        
        
        
        
        s0g1,s0g2,query1,query2,pos1,pos2,neg1,neg2 = self.query_graph_and_memory(rx_g[:,0,...])
        # h_en, state_en = self.encoder(x, init_state, supports) # B, T, N, hidden
        # hop_t = h_en[:, -1, :, :] # B, N, hidden (last state)        
        
        hop_t =self.encoder(x)        

        # if self.tod_embedding_dim > 0:
        #     tod = history_data[..., 1]
        # if self.dow_embedding_dim > 0:
        #     dow = history_data[..., 2]

        # Cvector = torch.zeros((batchsize,T_size,N_size,self.gatoutput),device=x.device)
        # gatoutlist = []
        # # for attn in self.attn_layers_s:
        # #     x = attn(hx,assist_params)
        # #     outlist.append(x)
        # #     hx = torch.cat([x,h],dim=-1)
        # for i,attn in enumerate(self.attn_layers_s):
        #     hx = torch.cat([Cvector,rx],dim=-1)
        #     gatoutput,Cvector = attn(hx)
        #     gatoutlist.append(gatoutput)
        # gatoutlist = torch.cat(gatoutlist,dim=-1) # B, T, N, F*dec_layers_num

        
        # h_att, query, pos, neg = self.query_memory(h_t)
        gate_adp_emb = self.adaptive_embedding_for_gate.expand(
            size=(batchsize, *self.adaptive_embedding_for_gate.shape)
        )
        sg1,sg2,lg1,lg2 = self.query_dyngraph(rx)
        # go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        go = x[:,-1,...]
        out = []
        # outhlist= []
        for t in range(self.horizon):
            # h_t = torch.cat([hop_t, rx[:,t,...]], dim=-1)
            
            h_de, hop_t = self.decoder.forward(torch.cat([go, y_cov[:, t, ...]], dim=-1), hop_t,rx[:,t,...],gate_adp_emb[:,t,...], [sg1[t],sg2[t]])
            # outhlist.append(h_de)
            go = self.proj(h_de)
            out.append(go)
            
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batch_seen):
                    go = labels[:, t, ...]
        # outhput = torch.stack(outhlist, dim=1) # B, T, N, F        
        output = torch.stack(out, dim=1) # B, T, N, D
        # gcrnoutputlist = self.gcrn_emb(output)
        ## (B, T, N, D+F*dec_layers_num) -> (B,N, T, Z) ->(B,N, T*Z) -> (B,N,T*output) -> (B,N,T,output) -> (B,T,N,output) 
        # finial_out = torch.cat([gcrnoutputlist,gatoutlist],dim=-1).transpose(1,2).reshape(batchsize,N_size,-1)
        # finial_out = self.output_proj2(self.gelu(self.output_proj1(finial_out))).reshape(batchsize,N_size,self.horizon,self.output_dim).transpose(1,2)
        
        # return output, h_att, query, pos, neg


        return {'prediction': output,"query1":query1,"query2":query2,"pos1":pos1,"pos2":pos2,"neg1":neg1,"neg2":neg2}
