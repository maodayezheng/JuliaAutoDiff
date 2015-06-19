#function DemoMNISTclassBinary()
    # Training a deep binary classifier on MNIST, with regularisation
    # The method uses Nesterov's accelerated gradient, with minibatches
    # (c) David Barber, University College London 2015

    # If running from repl within julia, need to first run:
    # using dbAutoDiff
    # include("DemoMNISTclass.jl);

    Ntrain=12000
    Ntest=2000
    BatchSize=500
    TrainingIts=300 # number of Nesterov updates

    include("loadmnist.jl")
    images,label=loadmnist()
    inds=find((label.==2) | (label.==3)) # just get 2s and 3s
    images=images[:,inds]; label=label[inds]

    r=randperm(size(images,2))
    images=images[:,r]; label=label[r]
    data=images[:,1:Ntrain]
    testdata=images[:,Ntrain+1:Ntrain+Ntest]
    classdata=-ones(1,length(r))
    for i=1:length(r)
        if label[i]==2 # classify as a 2 or non-2 digit
            classdata[i]=1
        end
    end
    classtraindata=(classdata[1:Ntrain])'
    classtestdata=(classdata[Ntrain+1:Ntrain+Ntest])'

    # Construct the DAG function:
    H=[784 250 100 50 20 1] # number of units in each layer
    L=length(H) # number of hidden layers
    # node indices:
    w=zeros(Int,L) # weight index
    h=zeros(Int,L) # hidden layer index (note that I call the input layer h[1])

    # nodes which are inputs have empty parent set
    # note that the ordering is irrelevant as long as this is a DAG
    # and that the scalar loss is the last node in the DAG
    nMAX=100 # maximum number of nodes in the DAG (alternatively, one could just push! new nodes rather than preallocating sufficient space)
    c=0 # node counter
    node=Array(ADnode,nMAX) # nodes
    node[xinput=h[1]=c+=1]=ADnode([]) # node has no parents
    node[class=c+=1]=ADnode([]) # node has no parents

    for i=2:L-1
        node[w[i]=c+=1]=ADnode([];returnderivative=true) # node is a parameter
        node[h[i]=c+=1]=ADnode([w[i] h[i-1]],FshiftedsigmaAx) # don't need derivatives of hidden layer
    end
    node[w[L]=c+=1]=ADnode([];returnderivative=true) # node is a parameter
    node[h[L]=c+=1]=ADnode([w[L] h[L-1]],FAx)

    # regulariser: if parameters are treated as separate nodes we need to deal with this in a special way. Fundamentally, we can do this in a basic way as:
    if false
        regnode=w[2:L]# which nodes are included in the regularisation term
        node[totalreg=c+=1]=ADnode(regnode[1],FmeanAbs)
        for i=2:length(regnode)
            node[thisreg=c+=1]=ADnode(regnode[i],FmeanAbs)
            node[totalreg=c+=1]=ADnode([totalreg thisreg],Fxpy)
        end
        node[reglambda=c+=1]=ADnode([])
        node[regpen=c+=1]=ADnode([totalreg reglambda],Fxy) # This is the final regularisation penalty
    end

    # alternatively, we can use a mapreduce style framework:
    #(totalreg,c)=mapreduce!(FmeanAbs,Fxpy,w[2:L],node)
    (totalreg,c)=mapreduce!(FmeanSquare,Fxpy,w[2:L],node)
    node[reglambda=c+=1]=ADnode([])
    node[regpen=c+=1]=ADnode([totalreg reglambda],Fxy) # This is the final regularisation penalty

    node[logloss=c+=1]=ADnode([class h[L]],FLogisticLoss)
    node[loss=c+=1]=ADnode([logloss regpen],Fxpy) # the loss we are minimising must be the final node in the graph.

    node=node[1:endnode(node)] # just take only the nodes required

    # instantiate parameter nodes and inputs:
    value=Array(Any,c) # function values on the nodes
    value[xinput]=data[:,1:BatchSize];
    value[class]=(classtraindata[1:BatchSize])'
    for i=2:L
        value[w[i]]=sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
    end
    value[reglambda]=0.05 # regularisation lambda

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
        value[xinput]=data[:,minibatch] # select batch
        value[class]=(classtraindata[minibatch])'

        # Nesterov Accelerated Gradient update:
        mu=1-3/(t+5);
        if t>10
            epsilon=0.01/(1+t/100)
        else
            epsilon=0.01
        end

        copy!(valueold,value)
        copyind!(oldnesterov,nesterov,parstoupdate)
        copyind!(value,value+mu*oldnesterov,parstoupdate)
        ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)
        copyind!(nesterov,mu*oldnesterov-epsilon*gradient,parstoupdate)
        copyind!(value,valueold+nesterov,parstoupdate)

        er[t]=value[loss]
        println("[$(t)]Loss = $(value[loss]),  learning rate=$epsilon")
    end
    toc()
    classout=ones(1,length(value[class]))
    classout[find(sigma(value[h[L]]).<0.5)]=-1
    classtrue=value[class]
    println("train accuracy = $(mean(classout.==classtrue))")

    # evaluate the test predictions:
    value[xinput]=testdata
    ADeval!(value,net;exclude=[class],doReverse=false) # net computes the loss (which depends on the class), so ignore all nodes that depend on the class
    classouttest=ones(size(value[h[L]]))
    classouttest[find(sigma(value[h[L]]).<0.5)]=-1
    println("test accuracy = $(mean(classouttest.==classtestdata))")

#end
