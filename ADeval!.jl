function ADeval!(value,net;auxvalue=[],gradient=[],returnall=true,exclude=[],doForward=true,doReverse=true)
#   calculate the value, auxiliary value and gradient of a net function
#   exclude is a collection of nodes to leave out of the forward pass calculation -- any descendents of these nodes are also automatically excluded
#= Note that there are some serious performance problems here. This is largely to do with
    garbage collection (suggested by ProfileView.view) and possibly also the issues here
    http://numericextensionsjl.readthedocs.org/en/latest/functors.html
    Unfortunately I've not been able to fix this whilst retaining fairly general code. This may need to wait until a future version of Julia that deals with shared memory more efficiently.
 =#
    node=net.node
    mess=net.message
    N=length(node)
    if isempty(auxvalue)
        auxvalue=Array(Any,N)
    end

    # forward pass:
    if doForward
        for i=setdiff(1:N,exclude)
            if length(findin(node[i].parents,exclude))>0
                push!(exclude,i)
            else            
                if !(node[i].input)
                    value[i],auxvalue[i]=node[i].f(value[node[i].parents]...)
                end
            end
        end
    end

    # reverse pass:
    
    if isempty(gradient)
        gradient=similar(value)
        for i=1:N-1
            if any(node[i].takederivative) & isa(value[i],Array)
                gradient[i]=zeros(size(value[i])) # preallocate
            else
                gradient[i]=0.0 # preallocate
            end
        end
    end

    if doReverse
        reverseorder=N:-1:1
        gradient[N]=1.0
        for i in reverseorder[2:N]
            if any(node[i].takederivative)
                count=0
                for c in node[i].children
                    count+=1
                    if count==1
                        gradient[i]=mess[i,c](value[node[c].parents]...,value[c],auxvalue[c],gradient[c])
                    else
                        gradient[i]=gradient[i]+mess[i,c](value[node[c].parents]...,value[c],auxvalue[c],gradient[c])
                    end
            end
            end
        end
    end
        
        if returnall
        return (value,gradient); # return the node values and total derivatives
    else
        tmp=gradient[find(map( (x)-> x.returnderivative,node))]
        if length(tmp)==1; tmp=tmp[1]; end
        return (value[N],tmp) # return the function and only the required derivatives
    end

end

