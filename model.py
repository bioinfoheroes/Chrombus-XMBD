class EdgeConv(MessagePassing):
    def __init__(self, n_heads, in_channels, out_channels):
        super().__init__(aggr='mean') #aggregation.
        self.n_heads = n_heads
        self.head_size = out_channels // n_heads
        self.all_heads = out_channels   
        self.softmax = Softmax(dim=-1)
        self.q = Linear(in_channels * 2, out_channels)
        self.k = Linear(in_channels * 2, out_channels)
        self.v = Linear(in_channels * 2, out_channels)
    def reshape(self, e):
        new_shape = e.size()[:-1] + (self.n_heads, self.head_size)
        e = e.view(*new_shape)
        return e.permute(1, 0, 2)  
    def forward(self, x, edge_index, edge_weight, batch=None):
        #print(f'number of edges: {edge_index.size(1)}')
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
    def message(self, x_i, x_j, edge_weight):
        query = self.q(torch.cat([x_i, x_j - x_i], dim=1))
        key = self.k(torch.cat([x_i, x_j - x_i], dim=1))
        value = self.v(torch.cat([x_i, x_j - x_i], dim=1))
        #print(f'values: {value.size()}')
        query = self.reshape(query)
        key = self.reshape(key)
        value = self.reshape(value)
        value = value * edge_weight[:,None]
        #print(f'keys: {key.size()}')
        scores = torch.mul(query, key)
        scores = scores / math.sqrt(self.head_size)
        probs = self.softmax(scores).to(device)  
        context = torch.mul(probs, value)
        context = context.permute(1, 0, 2).contiguous()
        context_shape = context.size()[:-2] + (self.all_heads, )
        context = context.view(*context_shape)
        #print(f'context: {context.size()}')
        return context

class DynamicEdgeConv(EdgeConv):
    def __init__(self, n_heads, in_channels, out_channels, thres=0.5, K=500, cis=8, n_neighbor=6):
        super().__init__(n_heads, in_channels, out_channels)
        self.K = K
        self.thres = torch.tensor(thres).to(device)
        self.cis =  torch.tensor(cis).to(device)
        self.n_neighbor = n_neighbor
    def forward(self, x, edge_index, batch):
        edge_kept = random.sample(range(0,edge_index.shape[1]), int(edge_index.shape[1] * 0.5))
        edge_index = edge_index[:,edge_kept]
        d = (edge_index[1] - edge_index[0]).abs()
        edge_weight = (d.log() - self.cis.log()).sign()
        return super().forward(x, edge_index, edge_weight=edge_weight)

class EdgeConvEncoder(torch.nn.Module): 
    def __init__(self, n_heads, in_channels, out_channels, cis_span, max_span):
        super().__init__()
        self.conv0 = DynamicEdgeConv(n_heads, in_channels, out_channels, thres=-.5, K=10000, cis=cis_span)
        self.conv1 = DynamicEdgeConv(n_heads, out_channels, out_channels // 2, thres=-.5, K=10000, cis=cis_span)
        self.conv2 = DynamicEdgeConv(n_heads, out_channels // 2, out_channels, thres=-.5, K=10000, cis=cis_span)
    def forward(self, x, edge_index, batch):
        pred_edge_index = getEdgeIndex(x.size(0), 128, max_span, batch=batch).to(device)
        x0 = self.conv0(x, pred_edge_index, batch)
        x0 = x0.relu()
        x = self.conv1(x0, pred_edge_index, batch)
        x = x.relu()       
        x = self.conv2(x, pred_edge_index, batch)
        x = x + x0  
        x = x.relu()    
        return x

def getEdgeIndex(N, n, max_span=64, batch=None):
    index = torch.zeros(2, 0, dtype=torch.long)
    index = index.to(device)
    for i in range(N):
        b = batch[i]
        j = torch.tensor([*range(i+1, min(n*(b+1), i+max_span+1))], dtype=torch.long) #without self-loop
        index = torch.cat([index, torch.tensor((numpy.repeat(i, len(j)), j)).to(device)], dim=1)        
    return to_undirected(index)
  


### Chrombus model
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cis_span = 9    # unit range of cis-interaction
max_span = 32   # 
n_heads = 8
in_channels, out_channels = 14, 32
model = GAE(encoder=EdgeConvEncoder(n_heads, in_channels, out_channels, cis_span, max_span), decoder=InnerProductDecoder())

