#function DemoMNIST()
    # Training a deep autoencoder on MNIST
    # The method uses Nesterov's accelerated gradient, with minibatches
    # David Barber, University College London 2015

    # If running from repl within julia, need to first run:
    # using dbAutoDiff, Winston
    # include("mnistmnibatch.jl); mnistminibatch()

    x=0.1*randn(5,10);

    #differentiate:   f(x)=sum(exp(-x.^2))

    # nodes which are inputs have empty parent set
    # note that the ordering is irrelevant as long as this is a DAG
    # and that the scalar function is the last node in the DAG
    nMAX=100 # maximum number of nodes in the DAG
    c=0 # node counter
    node=Array(ADnode,nMAX) # nodes
    node[c+=1]=ADnode([];returnderivative=true) # node has no parents
    node[c+=1]=ADnode(c,Fsquare)
    node[c+=1]=ADnode(c,Fnegative)
    node[c+=1]=ADnode(c,Fexp)
    node[c+=1]=ADnode(c,FmeanSquare)
    node=node[1:c] # just take only the nodes required

    # instantiate parameter nodes and inputs:
    value=Array(Any,c) # function values on the nodes
    value[1]=x

    (value,auxvalue,gradient,net)=compile(value,node); # compile the DAG and preallocate memory

    gradcheck(value,net,true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow
