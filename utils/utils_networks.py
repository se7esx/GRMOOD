import torch as th


class MLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, output_activation='relu', device='cpu', dtype=th.float64):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.num_hidden = num_hidden
        self.dtype = dtype
        
        self.linears = th.nn.ModuleList()
        if self.num_hidden == 0:
            self.linears.append(th.nn.Linear(input_dim, output_dim, device=device, dtype=dtype))
        elif self.num_hidden > 0:
            self.linears.append(th.nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype))
            
            for layer in range(num_hidden - 1):
                self.linears.append(th.nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype))
            self.linears.append(th.nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype))
        self.activation = th.nn.functional.relu
        if output_activation == 'relu':
            self.output_activation = th.nn.functional.relu
        elif output_activation == 'linear':
            self.output_activation = None
        else:
            raise 'unknown activation for output layer of MLP'
    def forward(self, x):
        if self.num_hidden == 0:
            x = self.linears[0](x)
            if not (self.output_activation is None):    
                x = self.output_activation(x)
        elif self.num_hidden > 0:
            # Pass the input tensor through each of our operations
            for layer in range(self.num_hidden):
                x = self.linears[layer](x)
                x = self.activation(x)
            x = self.linears[-1](x)
            if not (self.output_activation is None):    
                x = self.output_activation(x)
        return x


class MLP_dropout(th.nn.Module):
    """
    includes dropout on the computed distances 
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, dropout, skip_first=True, output_activation='relu', device='cpu', dtype=th.float64):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.num_hidden = num_hidden
        self.dtype = dtype
        self.skip_first = skip_first
        self.linears = th.nn.ModuleList()
        self.dropout_layers = th.nn.ModuleList()
        if not skip_first: # Will be used to apply dropout on the computed distances
            self.dropout_layers.append(th.nn.Dropout(dropout))
        
        if self.num_hidden == 0:
            self.linears.append(th.nn.Linear(input_dim, output_dim, device=device, dtype=dtype))
        elif self.num_hidden > 0:
            self.linears.append(th.nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype))
            for layer in range(num_hidden - 1):
                self.linears.append(th.nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype))
                self.dropout_layers.append(th.nn.Dropout(dropout))
            
            self.linears.append(th.nn.Linear(hidden_dim, output_dim).to(dtype))
            self.dropout_layers.append(th.nn.Dropout(dropout))
        

        self.activation = th.nn.functional.relu
        if output_activation == 'relu':
            self.output_activation = th.nn.functional.relu
        elif output_activation == 'linear':
            self.output_activation = None
        else:
            raise 'unknown activation for output layer of MLP'
    def forward(self, x):
        # Pass the input tensor through each of our operations
        if self.num_hidden == 0:
            if not self.skip_first:  # We apply dropout onto the dist features layer
                x = self.dropout_layers[0](x)
            x = self.linears[0](x)
            if not (self.output_activation is None):    
                x = self.output_activation(x)
            return x
        else:
            if not self.skip_first:
                x = self.dropout_layers[0](x)
            x = self.activation(self.linears[0](x))
            for layer in range(1, self.num_hidden):            
                x = self.dropout_layers[layer](x)
                x = self.activation(self.linears[layer](x))
            self.dropout_layers[-1](x)
            x = self.linears[-1](x)
            if not (self.output_activation is None):    
                x = self.output_activation(x)
        return x

class MLP_batchnorm(th.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, output_activation='relu',batchnorm_affine=True, device='cpu', dtype=th.float64):
        super().__init__()
        # Inputs to hidden layer linear transformation
        assert num_hidden > 0
        self.num_hidden = num_hidden
        self.dtype = dtype
        
        self.linears = th.nn.ModuleList()
        self.bns = th.nn.ModuleList()
        
        self.linears.append(th.nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype))
        self.bns.append(th.nn.BatchNorm1d(hidden_dim, affine=batchnorm_affine, device=device, dtype=dtype))
        
        for layer in range(num_hidden-1):
            self.linears.append(th.nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype))
            self.bns.append(th.nn.BatchNorm1d(hidden_dim, affine=batchnorm_affine, device=device, dtype=dtype))
        self.linears.append(th.nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype))
        self.activation = th.nn.functional.relu
        if output_activation == 'relu':
            self.bns.append(th.nn.BatchNorm1d(output_dim, affine=batchnorm_affine, device=device, dtype=dtype)) 
            self.output_activation = th.nn.functional.relu
        elif output_activation == 'linear':
            self.output_activation = None
        else:
            raise 'unknown activation for output layer of MLP'
    def forward(self, x):
        # Pass the input tensor through each of our operations
        
        for layer in range(self.num_hidden):
            x = self.linears[layer](x)
            x = self.bns[layer](x)
            x = self.activation(x)
            
        x = self.linears[-1](x)
        if not (self.output_activation) is None:
            x = self.bns[-1](x)
            x = self.output_activation(x)
        return x
    
