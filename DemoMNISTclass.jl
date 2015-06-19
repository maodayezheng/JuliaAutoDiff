#function DemoMNISTclass()
    # Training a deep classifier on MNIST
    # The method uses Nesterov's accelerated gradient, with minibatches
    # (c) David Barber, University College London 2015

    # If running from repl within julia, need to first run:
    # using dbAutoDiff, Winston
    # include("DemoMNISTclass.jl);

    Ntrain=2000
    BatchSize=1000
    TrainingIts=30 # number of Nesterov updates
    include("loadmnist.jl")
    images,label=loadmnist()
    r=randperm(size(images,2))
    data=images[:,r]
    class=zeros(10,70000)
    for i=1:70000
        class[label[i]+1,i]=10
    end
    class,dummy=Fsoftmax(class)
    class=class[:,r] # ensures that this is a distribution and bounded away from 0,1 (to avoid log problems)

    # Construct the DAG function:
    H=[784 250 100 50 10] # number of units in each layer
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
    node[xtrain=h[1]=c+=1]=ADnode([]) # node has no parents
    node[classtrain=c+=1]=ADnode([]) # node has no parents

    for i=2:L-1
        node[w[i]=c+=1]=ADnode([];returnderivative=true) # node is a parameter
        node[h[i]=c+=1]=ADnode([w[i] h[i-1]],FshiftedsigmaAx) # don't need derivatives of hidden layer
    end
    node[w[L]=c+=1]=ADnode([];returnderivative=true) # node is a parameter
    node[h[L]=c+=1]=ADnode([w[L] h[L-1]],FAx)
    node[loss=c+=1]=ADnode([classtrain h[L]],FKLsoftmax) # the loss we are minimising must be the final node in the graph.
    node=node[1:c] # just take only the nodes required

    # instantiate parameter nodes and inputs:
    value=Array(Any,c) # function values on the nodes
    value[xtrain]=data[:,1:BatchSize];
    value[classtrain]=class[:,1:BatchSize];
    for i=2:L
        value[w[i]]=sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
    end

    (value,auxvalue,gradient,net)=compile(value,node); # compile the DAG and preallocate memory

    #gradcheck(value,net,true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow

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
        value[xtrain]=data[:,minibatch] # select batch
        value[classtrain]=class[:,minibatch]

        # Nesterov Accelerated Gradient update:
        mu=1-3/(t+5);
        if t>10
            epsilon=0.1/(1+t/100)
        else
            epsilon=0.1
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

    classout=argcolmax(softmax(value[h[L]]))
    classtrue=argcolmax(value[classtrain])
    println("training accuracy = $(mean(classout==classtrue))")

#end
