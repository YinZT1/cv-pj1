import numpy as np
import copy

def affine_forward(x,w,b):
    out = None
    x_reshape = x.reshape(x.shape[0],-1)
    out = np.dot(x_reshape,w) + b
    return out,(x,w,b)

def affine_backward(dout,cache):
    x,w,b = cache
    dx,dw,db = None,None,None
    x_reshape = x.reshape(x.shape[0],-1)
    dx = np.dot(dout,w.T).reshape(x.shape)
    dw = np.dot(x_reshape.T,dout)
    db = np.sum(dout,axis=0)
    return dx,dw,db

def relu_forward(x):
    out = None
    out = np.maximum(x,0)
    cache = x
    return out,cache

def relu_backward(dout,cache):
    dx,x = None,cache
    temp = x>0
    dx = dout*temp
    return dx

def softmax_loss(x,y):
    N = x.shape[0]
    prob = np.divide(np.exp(x),np.sum(np.exp(x),axis=1,keepdims=True))
    log_probs = -np.log(prob)
    mat = np.zeros_like(log_probs)
    mat[np.arange(N),y] = 1
    loss = mat * log_probs
    loss = np.sum(loss) / N
    dx = prob
    dx[np.arange(N),y] -= 1
    dx = dx/N
    return loss,dx

class ThreeLayerModel():
    def __init__(self, input_dim = 3072, num_classes = 10 ,hidden_dim = 100, weight_scale = 1e-3, reg = 0.0):
        self.params = {}
        self.reg = reg
        self.ip_dim = input_dim
        self.params['W1'] = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self,X,y=None):
        '''
        if y not passed by, return value after X go through the whole nn.
        if y has a value, return loss & gradient dict

        - Returns: 
            output (y is None)
            loss,gradient dict(y is passed)
        '''
        scores = None
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        X = X.reshape(X.shape[0], -1)
        fwd1, cache_fwd1 = affine_forward(X, W1, b1)
        relu_out, cache_relu = relu_forward(fwd1)
        scores, cache_scores = affine_forward(relu_out, W2, b2)
        if y is None:
            return scores
        loss, grads = 0, {}
        loss,dz = softmax_loss(scores,y) # dz = dl / dscores
        loss = loss + self.reg * (np.sum(W1**2) + np.sum(W2**2))

        dx2,dW2,db2 = affine_backward(dz,cache_scores)
        grads['W2'] = dW2 + self.reg * W2 
        grads['b2'] = db2

        da = relu_backward(dx2,cache_relu)
        dx1, dW1, db1 = affine_backward(da, cache_fwd1)
        grads['W1'] = dW1 + self.reg * W1   
        grads['b1'] = db1

        return loss,grads
    

class multiple_layer_model():
    def __init__(self,input_dim = 3072,output_dim = 10,hidden_dim = [256,128], reg = 0.0, weight_scale = 1e-3):
        '''
        - input_dim: samples' dimension
        - output_dim: the number of classes
        - hidden_dim: a list containing each hidden layer's dimension. the length of which decide the number of hidden layers
        - reg: regularization factor
        - weight_scale: variance of initialization of weight matrix
        '''
        self.ipdim = input_dim
        self.opdim = output_dim
        self.reg = reg
        self.params = {}
        self.num_layers = len(hidden_dim)+2

        length = len(hidden_dim)
        self.params["W1"] = np.random.normal(loc=0, scale=weight_scale, size=(self.ipdim, hidden_dim[0]))
        self.params["b1"] = np.zeros(hidden_dim[0])
        assert length !=  1
        for i in range(2,length+1):
            self.params[f"W{i}"] = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim[i-2], hidden_dim[i-1]))
            self.params[f"b{i}"] = np.zeros(hidden_dim[i-1])
        self.params[f"W{i+1}"] = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim[-1],self.opdim))
        self.params[f"b{i+1}"] = np.zeros(self.opdim)  
        for k,v in self.params.items():
            self.params[k] = v.astype(np.float32)  

    def loss(self,X,y = None):
        '''
        if y not passed by, return value after X go through the whole nn.
        if y has a value, return loss & gradient dict

        - Returns: 
            output (y is None)
            loss,gradient dict(y is passed)
        '''
        X = X.astype(np.float32)
        h = X.reshape(X.shape[0],-1)
        scores = None
        hidden = {}
        relu_hidden = {}
        # forward ,calculating output
        for i in range(1,self.num_layers - 1):
            w = self.params[f"W{i}"]
            b = self.params[f"b{i}"]    
            h,cache_h = affine_forward(h,w,b)
            hidden[f"h{i}"] = h
            hidden[f"cache_h{i}"] = cache_h

            relu_out_hidden,cache_relu_hidden = relu_forward(h)
            relu_hidden[f"relu_h{i}"] = relu_out_hidden
            relu_hidden[f"relu_cache_h{i}"] = cache_relu_hidden

            h = relu_out_hidden
        
        w = self.params[f"W{i+1}"]
        b = self.params[f"b{i+1}"]
        scores,cache_scores = affine_forward(h,w,b)
        hidden[f"h{i+1}"] = scores
        hidden[f"cache_h{i+1}"] = cache_scores
        # for k,v in hidden.items():
        #     if k[:1] == "h":
        #         print(f"{k}:{v.shape}")
        #     else:
        #         print(f"{k}:    x:{v[0].shape},w:{v[1].shape},b:{v[2].shape}")

        #judge the function of loss
        #if y== None , return forward value

        if y is None:
            return scores
        
        #else , return loss and gradient
        loss = 0.0
        reg_loss = 0.0
        gradient = {}
        loss,dscores = softmax_loss(scores,y)
        for k in self.params.keys():
            if k[0] == "W":
                v = self.params[k]
                reg_loss += 0.5 * self.reg * np.sum(v * v)
        loss += reg_loss

        hidden[f"dh{i+1}"] = dscores
        dh = dscores
        for j in range(i+1,0,-1):
            h_cache = hidden[f"cache_h{j}"]
            dh,dw,db = affine_backward(dh,h_cache)
            hidden[f'dh{j}'] = dh
            hidden[f"dW{j}"] = dw
            hidden[f"db{j}"] = db

            if j!=1:
                da = relu_backward(dh,relu_hidden[f"relu_cache_h{j-1}"])
                dh = da
        list_dw = {key[1:]:val + self.reg * self.params[key[1:]]
                   for key,val in hidden.items() if key[:2] == "dW"}             
        list_db = {key[1:]:val for key,val in hidden.items() if key[:2] == 'db'}
        gradient.update(list_dw)
        gradient.update(list_db)
        return loss,gradient