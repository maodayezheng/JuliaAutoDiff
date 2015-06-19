#function DemoMNIST()
    # Training a deep autoencoder on MNIST
    # The method uses Nesterov's accelerated gradient, with minibatches
    # (c) David Barber, University College London 2015

    # If running from repl within julia, need to first run:
    # using dbAutoDiff, Winston
    # include("mnistmnibatch.jl); mnistminibatch()
    LogFile="LogFile"
    ParameterFile="parameters.jld"

    Ntrain=5000
    BatchSize=1000
    TrainingIts=500 # number of Nesterov updates
    include("loadmnist.jl")
    images,label=loadmnist()
    r=randperm(size(images,2))
    data=images[:,r]
    # bound away from 0 and 1 to avoid log(0) problems:
    tol=0.0001
    data[data.>(1-tol)]=1-tol
    data[data.<tol]=tol

    # Construct the DAG function:
    #H=[784 1000 500 250 30 250 500 1000 784] # number of units in each layer
    H=[784 250 100 50 100 250 784] # number of units in each layer
    L=length(H) # number of hidden layers
    # node indices:
    w=zeros(Int,L) # weight index
    h=zeros(Int,L) # hidden layer index (note that I call the input layer h[1])

    # nodes which are inputs have empty parent set
    # note that the ordering is irrelevant as long as this is a DAG
    # and that the scalar loss is the last node in the DAG
    nMAX=100 # maximum number of nodes in the DAG
    c=0 # node counter
    node=Array(ADnode,nMAX) # nodes
    node[ytrain=h[1]=c+=1]=ADnode([]) # node has no parents

    for i=2:L-1
        node[w[i]=c+=1]=ADnode([];returnderivative=true) # node is a parameter
        node[h[i]=c+=1]=ADnode([w[i] h[i-1]],FshiftedsigmaAx) # don't need derivatives of hidden layer
    end
    node[w[L]=c+=1]=ADnode([];returnderivative=true) # node is a parameter
    node[h[L]=c+=1]=ADnode([w[L] h[L-1]],FsigmaAx)

    node[sqerr=c+=1]=ADnode([ytrain h[L]],FmeanSquareLoss) # This won't form part of the objective function, but calculate this to monitor the square loss
    node[loss=c+=1]=ADnode([ytrain h[L]],FBinaryEntropyLoss) # the loss we are minimising must be the final node in the graph. We are going to minimise the KL loss (aka entropic loss, Bernoulli loss)
    node=node[1:c] # just take only the nodes required

    # instantiate parameter nodes and inputs:
    value=Array(Any,c) # function values on the nodes
    value[h[1]]=data[:,1:BatchSize];
    for i=2:L
        value[w[i]]=sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
    end

    (value,auxvalue,gradient,net)=compile(value,node); # compile the DAG and preallocate memory

    #gradcheck(value,net) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow

    # Nesterov Training:
    parstoupdate=find(map(x->x.returnderivative,net.node)) # node indices that are parameters
    er=zeros(TrainingIts)
    println("Batch Nesterov with decaying learning rate")
    tic()
    minibatchstart=1 # starting datapoint for the minibatch
    ADeval!(value,net; auxvalue=auxvalue,gradient=gradient) # ensure all values are computed. The named arguments are not necessary, but prevent garbage collection
    nesterov=0*gradient
    valueold=deepcopy(value)
    oldnesterov=deepcopy(nesterov)
    for t=1:TrainingIts
        if minibatchstart>Ntrain-BatchSize+1
            minibatchstart=Ntrain-BatchSize+1
        end
        minibatch=minibatchstart:minibatchstart+BatchSize-1
        minibatchstart=minibatchstart+BatchSize
        if minibatchstart>Ntrain
            minibatchstart=1
        end
        value[h[1]]=data[:,minibatch] # select batch

        # Nesterov Accelerated Gradient update:
        mu=1-3/(t+5);
        if t>10
            epsilon=1/(1+t/100)
        else
            epsilon=1
        end

        copy!(valueold,value)
        copyind!(oldnesterov,nesterov,parstoupdate)
        copyind!(value,value+mu*oldnesterov,parstoupdate)
        ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)
        copyind!(nesterov,mu*oldnesterov-epsilon*gradient,parstoupdate)
        copyind!(value,valueold+nesterov,parstoupdate)

        er[t]=value[loss]
        println("[$(t)]Loss = $(value[loss]), SqErrPerImage=$(784*value[sqerr]), learning rate=$epsilon")
        file=open(LogFile,"a")
        write(file,"[$(t)]Loss = $(value[loss]), SqErrPerImage=$(784*value[sqerr]), epsilon=$epsilon\n")
        close(file)

        # save intermittently to disk (julia currently has limited support for saving variables)
        if mod(t,10)==1 # save every 10th update
            ParFile=jldopen(ParameterFile,"w")
            write(ParFile,"v",value)
            close(ParFile)
        end
    end
    toc()

    colormap("grays")
    for i=1:20 # plot the reconstructions for a few datapoints
        p=imagesc([reshape(value[h[1]][:,i],28,28)'  reshape(value[h[L]][:,i],28,28)'],(0,1))
        display(p)
        println("press return key to continue")
        readline(STDIN)
    end

#end
