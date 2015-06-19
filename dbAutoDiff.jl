module dbAutoDiff

using NumericExtensions, MAT, Cartesian, HDF5, JLD #, ArrayViews

type ADnode
    parents::Array{Int,1} # node parent indices
    f::Function   # function that the node computes
    df::Array{Function,1}  # derivative function
    children::Array{Int,1} # node child indices
    takederivative # whether to take the derivative
    returnderivative::Bool # whether to return the derivative
    input::Bool # whether this is an input variable
    ADnode(parents,f=nx;returnderivative=false)=
        begin
            if isempty(parents)
                input=true
            else
                input=false
            end
            returnderivative=returnderivative==true
            takederivative=returnderivative
            if returnderivative & !input
                error("cannot return derivative for a node that has parents")
            end
            return new(collect(parents),f,Derivative[f],[0],takederivative,returnderivative,input)
        end
end


type network
    node::Array{ADnode,1}
    message::Array{Function,2}
    network(node,message)=new(node,message)
end


#import Base.eval

include("compile.jl")
include("ADeval!.jl")
include("defs.jl")
include("genFcode.jl")
include("genRcode.jl")
include("gradcheck.jl")


function endnode(node)
    d=falses(length(node))
    for i=1:length(d)
        d[i]=isdefined(node,i);
    end
    return last(find(d))
end
export endnode

function mapreduce!(f,op,nodeinds,node::Array{ADnode,1})
    # f is the function mapped onto each node in nodeinds
    # op is the binary reduction
    counter=endnode(node)
    node[redcounter=counter+=1]=ADnode(nodeinds[1],f)
    for i=2:length(nodeinds)
        node[thisred=counter+=1]=ADnode(nodeinds[i],f)
        node[redcounter=counter+=1]=ADnode([redcounter thisred],op)
    end
    return (redcounter,endnode(node))
end
export mapreduce!


#firstind(v)=begin; for i=1:length(v); isa(v[i],Tuple)? x[1] : x,v)
targ(x,a)= isa(x,Tuple)? x[a] : x # tuple argument
export firstind, targ

function copyind!(dest,source,ind)
    for i in ind
        copy!(dest[i],source[i])
    end
end
export copyind!

export ADnode, network, compile, ADeval!, genFcode, genRcode, gradcheck

export matread, jldopen
end
