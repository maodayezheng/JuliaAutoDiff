function DemoTiedNet()
    # Simple net to show how to compute the gradient and function value
    # Compares speed of calculating the gradient using ADeval! versus source code generation
    # David Barber, University College London 2015

    # To run from the Julia repl, need:
    # using dbAutoDiff
    # include("DemoTiedNet.jl"); DemoTiedNet()

    # Define the DAG:
    # note that the ordering is irrelevant as long as this is a DAG and the last node is the scalar loss
    L=10 # number of layers
    st="(y-NeuralNet($L layers, x input)).^2"
    println("Gradient descent for $(st)")
    node=Array(ADnode,4+2*L) # nodes
    c=0 # node counter
    node[ytrain=c+=1]=ADnode([]) # node is an input
    node[x=c+=1]=ADnode([]) # node is an input
    node[W=c+=1]=ADnode([];returnderivative=true) # node is an input
    node[c+=1]=ADnode([W x],FAx)
    node[c+=1]=ADnode(c-1,Fsigma)
    for l=2:L
        node[c+=1]=ADnode([W c-1],FAx) # all layers have the same tied weight
        node[c+=1]=ADnode(c-1,Fsigma)
    end
    node[sqerr=c+=1]=ADnode([c-1 ytrain],FmeanSquareLoss)

    # Define a fake training problem (just random data)
    Ntrain=10000
    X=50; Y=50; H=50
    xval=sign(randn(X,Ntrain))
    Wval=sign(randn(H,X))/sqrt(H)
    value=Array(Any,4+2*L) # function values on the nodes
    value[x]=xval; value[W]=Wval;  value[ytrain]=xval
    (value,auxvalue,gradient,net)=compile(value,node) # compile and preallocate memory
    #gradcheck(value,net)


    # generate source code for the function and its derivative (the Julia compiler seems to be able to optimise these a bit better than using the ADeval! function)
    # Note that this produces the files NN10layers!.jl and gradNN10layers!.jl. Both functions are inplace -- that is they don't return anything, simply updating value in the first case and gradient in the second. The first updates all the values in a forward pass and the second the gradient only as a function of the forward pass (so we may need to run the forward pass before the gradient to ensure the values are uptodate for the gradient).
    fname="NN10layers"
    genFcode(value,auxvalue,node,fname)
    genRcode(value,auxvalue,gradient,net.node,net.message,"grad"*fname)
    include(fname*"!.jl")
    include("grad"*fname*"!.jl")

    grad=:(grad(value,auxvalue,gradient))
    grad.head=:call
    grad.args[1]=symbol("grad"*fname*"!")
    grad.args[2]=value
    grad.args[3]=auxvalue
    grad.args[4]=gradient

    fun=:(fun(value,auxvalue))
    fun.head=:call
    fun.args[1]=symbol(fname*"!");
    fun.args[2]=value
    fun.args[3]=auxvalue

    #warm start:
    eval(fun)
    eval(grad)
    ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)

    Loop=50
    println("Using generated source code:")
    tic()
    # note : to calculate the gradient, this assumes the values are uptodate. Otherwise need a function call
    for i=1:Loop
        eval(fun)
        eval(grad)
    end
    toc()

    println("using ADeval!:")
    tic()
    for i=1:Loop
        ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)
    end
    toc()

    println("using ADeval! (no named arguments):")
    tic()
    for i=1:Loop
        ADeval!(value,net)
    end
    toc()
end
