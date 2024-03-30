
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.optimize as op
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

def split_data(data,
               test_fraction, 
               validation_fraction):

    # Split data into a part for training and a part for testing
    train_data, test_data = train_test_split(data, 
                                         test_size=test_fraction, 
                                         shuffle=True)

    # Split the training data into a part for training (fitting) and
    # a part for validating the training.
    v_fraction = validation_fraction * len(data) / len(train_data)
    train_data, valid_data = train_test_split(train_data, 
                                          test_size=v_fraction,
                                          shuffle=True)

    # reset the indices in the dataframes and drop the old ones
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)
    
    return train_data, valid_data, test_data 

def split_source_target(df, source, target):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    x = np.array(df[source])
    t = np.array(df[target])
    return x, t

# return a batch of data for the next step in minimization
def get_batch(x, t, batch_size):
    # selects at random "batch_size" integers from 
    # the range [0, batch_size-1] with replacement
    # corresponding to the row indices of the training 
    # data to be used
    rows = torch.randint(0, len(x)-1, size=(batch_size,))
    return x[rows], t[rows]

# Note: there are several average loss functions available 
# in pytorch, but it's useful to know how to create your own.
def average_quadratic_loss(f, t, x=None):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)

def average_cross_entropy_loss(f, t, x=None):
    # f and t must be of the same shape
    loss = torch.where(t > 0.5, torch.log(f), torch.log(1 - f))
    return -torch.mean(loss)

def average_quantile_loss(f, t, x):
    # f and t must be of the same shape
    tau = x.T[-1] # last column is tau.
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))

# function to validate model during training.
def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode

    with torch.no_grad(): # no need to compute gradients wrt. x and 
        # remember to reshape!
        outputs = model(inputs).reshape(targets.shape)
    return avloss(outputs, targets, inputs)
        
def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def train(model, optimizer, dictfile, early_stopping_count,
          avloss, getbatch,
          train_data, valid_data, 
          features, target,
          batch_size,
          n_iterations, 
          traces, 
          step=10, 
          change=0.005):
    
    train_x, train_t = split_source_target(train_data, 
                                           features, target)
    
    valid_x, valid_t = split_source_target(valid_data, 
                                           features, target)

    train_x, train_t = split_source_target(train_data, features, target)
    
    valid_x, valid_t = split_source_target(valid_data, features, target)

    # load data onto computational device
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        train_x = torch.from_numpy(train_x).float().to(device)
        train_t = torch.from_numpy(train_t).float().to(device)
    
        valid_x = torch.from_numpy(valid_x).float().to(device)
        valid_t = torch.from_numpy(valid_t).float().to(device)
    
    
    # to keep track of average losses
    xx, yy_t, yy_v = traces

    # place model on current computational device
    model = model.to(device)
    
    # save model with smallest validation loss
    # if after early_stopping_count iterations 
    # no validation scores are lower than the
    # current lowest value.
    min_acc_v = 1.e30
    stopping_count = 0
    jjsaved = 0
    
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("%9s %9s %9s" % \
          ('iteration', 'train-set', 'valid-set'))
    
    for ii in range(n_iterations):
                
        stopping_count += 1
            
        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        x, t = getbatch(train_x, train_t, batch_size)
        
        # compute the output of the model for the batch of data x
        # Note: outputs is 
        #   of shape (-1, 1), but the tensor targets, t, is
        #   of shape (-1,)
        # for the tensor operations with outputs and t to work
        # correctly, it is necessary that they be of the same
        # shape. We can do this with the reshape method.
        outputs = model(x).reshape(t.shape)
        
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t, x)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]).item() 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n]).item()
            print(f'\r{ii}',end='')

            if acc_v < (1-change)*min_acc_v:
                min_acc_v = acc_v
                torch.save(model.state_dict(), dictfile)
                stopping_count = 0
                jjsaved = ii
            else:
                if stopping_count > early_stopping_count:
                    print('\n\nstopping early!')
                    break
                    
            if len(xx) < 1:
                xx.append(0)
                print("%9d %9.7f %9.7f" % (xx[-1], acc_t, acc_v))
            elif len(xx) < 5:
                xx.append(xx[-1] + step)
                print("%9d %9.7f %9.7f" % (xx[-1], acc_t, acc_v))
            else:
                xx.append(xx[-1] + step)
                saved = ' %9d: %9d/%10.8f/%9d' % \
                (ii, jjsaved, min_acc_v, stopping_count)
                print("\r%9d %9.7f %9.7f%s" % \
                      (xx[-1], acc_t, acc_v, saved), end='')
                
            yy_t.append(acc_t)
            yy_v.append(acc_v)
                
    print()
    return (xx, yy_t, yy_v)

def plot_average_loss(traces, ftsize=18):
    
    xx, yy_t, yy_v = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel('Iterations', fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')

    plt.show()
    
def hist_data(df, data):

    xbins, xmin, xmax = data.alpha_bins, data.alpha_min, data.alpha_max
    ybins, ymin, ymax = data.beta_bins,  data.beta_min,  data.beta_max
    
    xrange = (xmin, xmax)
    yrange = (ymin, ymax)
        
    # weighted histogram   (count the number of ones per bin)
    hw, xedges, yedges = np.histogram2d(df.alpha, df.beta, 
                                        bins=(xbins, ybins), 
                                        range=(xrange, yrange), 
                                        weights=df.Z0)

    # unweighted histogram (count number of ones and zeros per bin)
    hu, xedges, yedges = np.histogram2d(df.alpha, df.beta, 
                                        bins=(xbins, ybins), 
                                        range=(xrange, yrange)) 

    p =  hw / (hu + 1.e-10)    
    
    return p, xedges, yedges

def nll(params, *args):
    a, b = params
    dnn, d = args
    return dnn(a, b, d)

def best_fit(nll, dnn, d):
    # ----------------------------------------------
    # find best-fit value
    # ----------------------------------------------
    guess   = [0.5, 0.5]
    results = op.minimize(nll, guess, args=(dnn, d), 
                          method='Nelder-Mead')
    if results.success:
        alpha, beta = results.x
        print('alpha: %10.3f, beta: %10.3f*%-6.1e, min(fun): %10.3f' %\
        (alpha*d.alpha_scale, beta, d.beta_scale, results.fun))
        print('%19sbeta:   %10.5f' % ('', beta*d.beta_scale))
    return results

def plot_model(df, d, dnn=None, results=None, hist=None,
                          filename='fig_pvalue_model.png',
                          fgsize=(5, 5), ftsize=18):
        
    # approximate probability via histogramming
    # P:  bin contents as a 2d (xbins, ybins) array
    # xe: bin boundaries in x
    # ye: bin boundaries in y
    if hist:
        P, xe, ye = hist
    else:
        P, xe, ye = hist_data(df, d)
    
    xbins,ybins= P.shape
    xmin, xmax = xe.min(), xe.max()
    ymin, ymax = ye.min(), ye.max()
    
    # flatten arrays so that p, x, and y are 1d arrays
    # of the same length.    
    # get bin centers
    x   = (xe[1:] + xe[:-1])/2
    y   = (ye[1:] + ye[:-1])/2
    X,Y = np.meshgrid(x, y)
    x   = X.flatten()
    y   = Y.flatten()
    
    # WARNING: must transpose P so that X, Y, and P have the
    # same shape
    P   = P.T
    p   = P.flatten()
                
    # Now make plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fgsize)
    
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(r'$\alpha$', fontsize=ftsize)
    
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.set_ylabel(r'$\beta$',  fontsize=ftsize)
    
    mylevels = np.array([0.68, 0.80, 0.90, 0.95])

    colormap = 'rainbow'
    
    cs = ax.contour(X, Y, P, 
               extent=(xmin, xmax, ymin, ymax),
               levels=mylevels,
               linewidths=2,
               linestyles='dashed',
               cmap=colormap)
    
    # ----------------------------------------------
    # compute model output at every grid point
    # then reshape to a 2d array
    # ----------------------------------------------
    if dnn:
        xbins,ybins= 50, 50        
        xstep = (xmax - xmin)/xbins
        ystep = (ymax - ymin)/ybins
        x     = np.arange(xmin, xmax, xstep) + xstep/2
        y     = np.arange(ymin, ymax, ystep) + ystep/2
        X, Y  = np.meshgrid(x, y)
        x     = X.flatten()
        y     = Y.flatten()
        F     = dnn(x, y, d).reshape(X.shape)

        cs = ax.contour(X, Y, F, 
                    extent=(xmin, xmax, ymin, ymax),
                    levels=mylevels, 
                    linewidths=2,
                    cmap=colormap) 

    ax.clabel(cs, cs.levels, inline=True, 
              fontsize=18, fmt='%4.2f', 
              colors='black')
    
    if results:
        if results.success:
            alpha, beta = results.x
            ax.scatter(alpha, beta, s=80, c='black')
        
            beta *= d.beta_scale
            xpos = xmin + 0.1*(xmax-xmin)
            ypos = ymin + 0.9*(ymax-ymin)
            ax.text(xpos, ypos, 
                r'$\alpha: %5.3f$\,\,$\beta: %7.5f$' % \
                (alpha, beta), fontsize=ftsize)
        
    ax.grid()

    plt.tight_layout()
    
    if filename != None:
        print('saved to file:', filename)
        plt.savefig(filename)
        
    plt.show()
