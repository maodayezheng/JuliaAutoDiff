function compile(value,node::Array{ADnode,1})

    println("Compiling into a tree and storing messages:")
    N=length(node)

    auxvalue=Array(Any,N)
    # get the computation tree:
    G=zeros(Bool,N,N)
    returnderivative=zeros(Bool,N)
    for i=1:N
        G[i,node[i].parents]=1
    end
    for i=1:N
        node[i].children=setdiff(find(G[:,i]),i)
    end

    # assume for now that the nodes are defined in ancestral order
    # forward pass:
    println("Forward Pass compilation:")
    for i=1:N
        print("node[$i]")
        if !(node[i].input)
            node[i].takederivative=map( (x)-> any(x.takederivative), node[node[i].parents] ) # set to true for those parents that require derivative
            value[i],auxvalue[i]=node[i].f(value[node[i].parents]...)
        end
        println(": ok")
    end
    gradient=similar(value)
    for i=1:N-1
        if any(node[i].takederivative) & isa(value[i],Array)
            gradient[i]=zeros(size(value[i])) # preallocate
        else
            gradient[i]=0.0 # preallocate
        end
    end
    gradient[N]=1.0

    # get the message functions:
    println("Storing messages:")
    mess=Array(Function,N,N)
    for i=1:N
        if !(node[i].input)
            print("node $i:")
            pars=node[i].parents
            println("has parents $pars")
            for j in pars
                mess[j,i]=node[i].df[findfirst(pars,j)];
            end
        end
    end
    net=network(node,mess)
    return (value,auxvalue,gradient,net)

end
