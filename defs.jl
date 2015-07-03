# (c) David Barber, University College London 2015
#= How this works:

GENERAL AUTODIFF THEORY:

According to the general autodiff theory (see my online notes http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/ParameterTying.pdf) the total derivative t[n] at a node n in the graph is related to the children c of node n by
 \sum_c t[c] df[c]/df[n]

In order to make this efficient in dealing with the case that the nodes may represent non-scalar quantities, it is useful to define a derivative as that returns a function of the child total derivative

f(t) = \sum_c t[c] df[c]/df[n]

For example, for the scalar function f(x,A,y)=x'*A*y, we could return a derivative tuple (A*y,x*y',A'*x) with respect to each of the arguments.


RETURNING A TOTAL DERIVATIVE FUNCTION:

During the reverse pass, this gradient will be multiplied by t_c (which will be a scalar). We therefore return the function (t.*(A*y),t.*(x*y'), t.*(A'*x))

The rationale behind this is perhaps more clear when considering a node that computes h=sigma(W*x).  Since the output of this node h is a vector, in theory we need to calculate all the derivatives from these elements of this child h to its parents W and x. For example,

dh[i]/dW[a,b] = dsigma(W[a,:]'*x)*x[b]*delta[i,a]

where delta[i,a] is the Kronecker delta function and dsigma is the derivative of the transfer function sigma. However, when we use this as part of the reverse recursion, the Kronecker delta gives a simplification, namely

\sum_i t[i]dh[i]/dW[a,b] = t[a]dsigma(W[a,:]'*x)*x[b]

What this means is that we can exploit the mathematical simplication that results from the Kronecker delta and return instead a derivative as a function of the child total derivative t.


REUSING INFORMATION BY STORING AUXILIARY VALUES:

Some computational savings can be made in the reverse pass by reusing results computed during the forward pass.

For example, for the logisitic sigmoid, we calculate f=sigma(W*h) on the forward pass. The derivative pass requires us to calculate dsigma(W*x) but since this is equal to sigma(W*h)*(1-sigma(W*h)) this is just a simple function of the forward pass result, namely f.*(1-f) so that we can simply reuse the forward calculation values, without having to compute the sigmoid function again in the reverse pass -- this saves on computation.

The code therefore allows during the forward pass to store auxilliary information at a node that might be useful for speeding up the computation in the backward pass. The deriviate can then be a function of the forward pass value (self) or the auxilliary information (aux) stored at the forward pass node.


BAD PROGRAMMING (!):

I should really write a macro to help with the derivative definitions below, since the form for each function is quite standard.

Note that the awkward way the deriviatives are defined is to ensure that they are generic functions (rather than anonymous functions) which both improves performance and also makes it possible to produce explicit source code (genFcode, genRcode), if required. In the current version of Julia 0.3.7, I don't see any other way to do this more elegantly, but hopefully this can be improved in a future version.

=#

# function definitions:
    function sigma!(st,x::Array{Float64,2})
    for i=1:length(x)
        @inbounds st[i]=1/(1+exp(-x[i]))
    end
    st
end

sigma(x)=sigma!(Array(fptype(eltype(x)),size(x)),x)

function sigma!(st,x::Array{Float64,1})
    for i=1:length(x)
        @inbounds st[i]=1/(1+exp(-x[i]))
    end
    st
end

sigma(x)=sigma!(Array(fptype(eltype(x)),size(x)),x)
export sigma


function argcolmax(x)
    out=Array(Int,size(x,2))
    for i=1:length(out)
        out[i]=indmax(x[:,i])
    end
    return out
end
export argcolmax

softmax(x::Array{Float64,2})=exp(x)./sum(exp(x),1)
export softmax



# Derivatives:

# functions are defined as F(x)=(self,aux). Here self is the function value and aux is the auxiliary value (it can be empty []) that might be useful to speed up the return pass calculation.

# DF[i] is the derivative function of F[x1,x2,...] with respect to its argument xi, as a function of the child total derivative t

Derivative=Dict() # mappings of functions to their derivatives

nx(x)=(nothing,[]) # ignore this
Dnx=Array(Function,1)
Dnx[1]=nx
Derivative[nx]=Dnx
export nx

Flinear(x)=(x,[])
Dlinear=Array(Function,1)
Dlinear[1]=dx1(x,self,aux,t)=t.*ones(size(x))
Dlinear[1]=dx1(x::Float64,self,aux,t)=t
Derivative[Flinear]=Dlinear # Define dictionary lookup
export Flinear
export dx1 # need for source code execution


Fnegative(x)=(-x,[])
Dnegative=Array(Function,1)
Dnegative[1]=dnegative1(x,self,aux,t)=-t.*ones(size(x))
Dnegative[1]=dnegative1(x::Float64,self,aux,t)=-t
Derivative[Fnegative]=Dnegative # Define dictionary lookup
export Fnegative
export dnegative1 # need for source code execution


Fsquare(x)=(x.^2,[])
Dsquare=Array(Function,1)
Dsquare[1]=dxx1(x,self,aux,t)=2.0*t.*x
Derivative[Fsquare]=Dsquare # Define dictionary lookup
export Fsquare
export dxx1 # need for source code execution

Fsum(x)=(sum(x),[])
Dsum=Array(Function,1)
Dsum[1]=dsum1(x,self,aux,t)=t
Derivative[Fsum]=Dsum # Define dictionary lookup
export Fsum
export dsum1 # need for source code execution

Flog(x)=(log(x),[])
Dlog=Array(Function,1)
Dlog[1]=dlog1(x,self,aux,t)=t./x
Derivative[Flog]=Dlog
export Flog
export dlog1 # need for source code execution

Fexp(x)=(exp(x),[])
Dexp=Array(Function,1)
Dexp[1]=dexp1(x,self,aux,t)=t.*self
Derivative[Fexp]=Dexp
export Fexp
export dexp1# need for source code execution


FxtAy(x,A,y)=(x'*A*y,[]) # works for vectors x,y and matrix A
DxtAy=Array(Function,3)
DxtAy[1]=dxtAy1(x,A,y,self,aux,t)=t.*(A*y) # derivative wrt x (arg 1)
DxtAy[2]=dxtAy2(x,A,y,self,aux,t)=t.*(x*y') # derivative wrt A (arg 2)
DxtAy[3]=dxtAy3(x,A,y,seld,aux,t)=t.*(A'*x) # derivative wrt y (arg 3)
Derivative[FxtAy]=DxtAy
export FxtAy
export DxtAy1,DxtAy2,Dxty3 # need for source code execution

Fxy(x,y)=(x.*y,[])
Dxy=Array(Function,2)
Dxy[1]=dty(x,y,self,aux,t)=t.*y
Dxy[2]=dtx(x,y,self,aux,t)=t.*x
Derivative[Fxy]=Dxy
export Fxy
export dty,dtx # need for source code execution

FAx(A,x)=(A*x,[])
DAx=Array(Function,2)
DAx[1]=DAx1(A,x,s,a,t)=t*x'
DAx[2]=DAx2(A,x,s,a,t)=A'*t
Derivative[FAx]=DAx
export FAx,DAx
export DAx1, DAx2 # need for source code execution

Fxpy(x,y)=(x.+y,[])
Dxpy=Array(Function,2)
Dxpy[1]=Dxpy1(x,y,s,a,t)=t.*ones(size(x))
Dxpy[2]=Dxpy2(x,y,s,a,t)=t.*ones(size(y))
Dxpy[1]=Dxpy1(x::Float64,y::Float64,s,a,t)=t
Dxpy[2]=Dxpy2(x::Float64,y::Float64,s,a,t)=t
Derivative[Fxpy]=Dxpy
export Fxpy,Dxpy
export Dxpy1, Dxpy2 # need for source code execution

Fxpypz(x,y,z)=(x.+y.+z,[])
Dxpypz=Array(Function,3)
Dxpypz[1]=Dxpypz1(x,y,z,s,a,t)=t.*ones(size(x))
Dxpypz[2]=Dxpypz2(x,y,z,s,a,t)=t.*ones(size(y))
Dxpypz[3]=Dxpypz3(x,y,z,s,a,t)=t.*ones(size(z))
Dxpypz[1]=Dxpypz1(x::Float64,y::Float64,z::Float64,s,a,t)=t
Dxpypz[2]=Dxpypz2(x::Float64,y::Float64,z::Float64,s,a,t)=t
Dxpypz[3]=Dxpypz3(x::Float64,y::Float64,z::Float64,s,a,t)=t
Derivative[Fxpypz]=Dxpypz
export Fxpypz,Dxpypz
export Dxpypz1, Dxpypz2, Dxpypz3 # need for source code execution

Fsigma(x::Array{Float64,2})=(sigma(x),[]);
Dsigma=Array(Function,1)
Dsigma[1]=Dsigma1(x::Array{Float64,2},self,aux,t::Array{Float64,2})=t.*self.*(1.-self)
Derivative[Fsigma]=Dsigma
export Fsigma
export Dsigma1 # need for source code execution


Ftanh(x::Array{Float64,2})=(tanh(x),[]);
Dtanh=Array(Function,1)
Dtanh[1]=Dtanh1(x::Array{Float64,2},self,aux,t::Array{Float64,2})=t.*(1-self.^2)
Derivative[Ftanh]=Dtanh
export Ftanh
export Dtanh1 # need for source code execution

# rectified linear
Frectlin(x::Array{Float64,2})=(max(x,0),[]);
Drectlin=Array(Function,1)
Drectlin[1]=Drectlin1(x::Array{Float64,2},self,aux,t::Array{Float64,2})=t.*(x.>0)
Derivative[Frectlin]=Drectlin
export Frectlin
export Drectlin1 # need for source code execution


# Neural Net layer
FsigmaAx(A::Array{Float64,2},x::Array{Float64,2})=(sigma(A*x),[])
DsigmaAx=Array(Function,2)
DsigmaAx[1]=DsigmaAx1(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=(t.*self.*(1.-self))*x'
DsigmaAx[2]=DsigmaAx2(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=A'*(t.*self.*(1.-self))
Derivative[FsigmaAx]=DsigmaAx
export FsigmaAx
export DsigmaAx1, DsigmaAx2 # need for source code execution

FshiftedsigmaAx(A::Array{Float64,2},x::Array{Float64,2})=begin a=sigma(A*x); return (5*a-2.5,5.*a.*(1.-a)); end
DshiftedsigmaAx=Array(Function,2)
DshiftedsigmaAx[1]=DshiftedsigmaAx1(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=(t.*aux)*x'
DshiftedsigmaAx[2]=DshiftedsigmaAx2(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=A'*(t.*aux)
Derivative[FshiftedsigmaAx]=DshiftedsigmaAx

FshiftedsigmaAx(A,x)=begin a=sigma(A*x); return (5*a-2.5,5.*a.*(1.-a)); end
DshiftedsigmaAx=Array(Function,2)
DshiftedsigmaAx[1]=DshiftedsigmaAx1(A,x,self,aux,t)=(t.*aux)*x'
DshiftedsigmaAx[2]=DshiftedsigmaAx2(A,x,self,aux,t)=A'*(t.*aux)
Derivative[FshiftedsigmaAx]=DshiftedsigmaAx
export FshiftedsigmaAx
export DshiftedsigmaAx1, DshiftedsigmaAx2 # need for source code execution

# Leon Bottou scaled tanh
FbTanhAx(A::Array{Float64,2},x::Array{Float64,2})=begin a=tanh(0.6666*A*x); return (1.7159*a,1.7159*0.6666*(1.-a.*a)); end
DbTanhAx=Array(Function,2)
DbTanhAx[1]=DbTanhAx1(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=(t.*aux)*x'
DbTanhAx[2]=DbTanhAx2(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=A'*(t.*aux)
Derivative[FbTanhAx]=DbTanhAx
export FbTanhAx, DbTanhAx1, DbTanhAx2

FtanhAxo2(A,x)=begin a=sigma(A*x); return (2*a-1.,2.*a.*(1.-a)); end
DtanhAxo2=Array(Function,2)
DtanhAxo2[1]=DtanhAxo21(A,x,self,aux,t)=(t.*aux)*x'
DtanhAxo2[2]=DtanhAxo22(A,x,self,aux,t)=A'*(t.*aux)
Derivative[FtanhAxo2]=DtanhAxo2
export FtanhAxo2
export DtanhAxo21, DtanhAxo22 # need for source code execution

# rectifiedlinear(A*x)
FrectlinAx(A::Array{Float64,2},x::Array{Float64,2})=begin a=A*x; return (max(a,0.),(a.>0));end
DrectlinAx=Array(Function,2)
DrectlinAx[1]=DrectlinAx1(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=(t.*aux)*x'
DrectlinAx[2]=DrectlinAx2(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=A'*(t.*aux)
Derivative[FrectlinAx]=DrectlinAx
export FrectlinAx
export DrectlinAx1, DrectlinAx2 # need for source code execution

# sqrt(2)*rectifiedlinear(A*x) (scaled version)
FsrectlinAx(A::Array{Float64,2},x::Array{Float64,2})=begin a=1.414*A*x; return (max(a,0.),(a.>0));end
DsrectlinAx=Array(Function,2)
DsrectlinAx[1]=DsrectlinAx1(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=(t.*aux)*x'
DsrectlinAx[2]=DsrectlinAx2(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=A'*(t.*aux)
Derivative[FsrectlinAx]=DsrectlinAx
export FsrectlinAx
export DsrectlinAx1, DsrectlinAx2 # need for source code execution


# This is an alternative sigmoid 2.5*A*x/(1+abs(A*x)):
FssigmaAx(A::Array{Float64,2},x::Array{Float64,2})=begin a=A*x; return (2.5*a./(1.+abs(a)),2.5./(1.+abs(a)).^2); end
DssigmaAx=Array(Function,2)
DssigmaAx[1]=DssigmaAx1(A::Array{Float64,2},x::Array{Float64,2},s,aux,t::Array{Float64,2})=(t.*aux)*x'
DssigmaAx[2]=DssigmaAx2(A::Array{Float64,2},x::Array{Float64,2},s,aux,t::Array{Float64,2})=A'*(t.*aux)
Derivative[FssigmaAx]=DssigmaAx
export FssigmaAx
export DssigmaAx1, DssigmaAx2 # need for source code execution


FfastsigmaAx(A::Array{Float64,2},x::Array{Float64,2})=begin a=A*x; return (0.5+0.5*a./(1.+abs(a)),0.5./(1.+abs(a)).^2); end
DfastsigmaAx=Array(Function,2)
DfastsigmaAx[1]=DfastsigmaAx1(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=(t.*aux)*x'
DfastsigmaAx[2]=DfastsigmaAx2(A::Array{Float64,2},x::Array{Float64,2},self,aux,t::Array{Float64,2})=A'*(t.*aux)
Derivative[FfastsigmaAx]=DfastsigmaAx
export FfastsigmaAx
export DfastsigmaAx1, DfastsigmaAx2 # need for source code execution

# biases in a standard sigmoid:
FsigmaAxBias(A,x,b)=(1./(1+exp(-(A*x.+b))),[])
DsigmaAxBias=Array(Function,3)
DsigmaAxBias[1]=DsigmaAxBias1(A,x,b,self,aux,t)=(t.*self.*(1.-self))*x'
DsigmaAxBias[2]=DsigmaAxBias2(A,x,b,self,aux,t)=W'*(t.*self.*(1.-self))
DsigmaAxBias[3]=DsigmaAxBias3(A,x,b,self,aux,t)=sum(t.*self.*(1.-self),2)
Derivative[FsigmaAxBias]=DsigmaAxBias
export FsigmaAxBias
export DsigmaAxBias1, DsigmaAzBias2, DsigmaAxBias3 # need for source code execution


# mean square: useful for squared regularisation term
FmeanSquare(x::Array{Float64,2})=begin DN=prod(size(x)); return ((vnorm(x,2))^2/DN,[]); end
DmeanSquare=Array(Function,1)
DmeanSquare[1]=DmeanSquare1(x,self,aux,t)=begin DN=prod(size(x)); return  2*t*x/DN; end
Derivative[FmeanSquare]=DmeanSquare
export FmeanSquare
export DmeanSquare1 # need for source code execution

# mean abs: useful for L1 regularisation term
FmeanAbs(x::Array{Float64,2})=begin DN=prod(size(x)); return ((vnorm(x,1))/DN,[]); end
DmeanAbs=Array(Function,1)
DmeanAbs[1]=DmeanAbs1(x,self,aux,t)=begin DN=prod(size(x)); return  t*sign(x)/DN; end
Derivative[FmeanAbs]=DmeanAbs
export FmeanAbs
export DmeanAbs1 # need for source code execution

# mean square loss:
FmeanSquareLoss(x::Array{Float64,2},y::Array{Float64,2})=begin DN=prod(size(x)); d=x-y; return ((vnorm(d,2))^2/DN,2*d/DN); end
DmeanSquareLoss=Array(Function,2)
DmeanSquareLoss[1]=dmeanSquareLoss1(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Array{Float64,2})=t*aux
DmeanSquareLoss[2]=dmeanSquareLoss2(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Array{Float64,2})=-t*aux
DmeanSquareLoss[1]=dmeanSquareLoss1(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Float64)=t*aux
DmeanSquareLoss[2]=dmeanSquareLoss2(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Float64)=-t*aux
Derivative[FmeanSquareLoss]=DmeanSquareLoss
export FmeanSquareLoss
export dmeanSquareLoss1, dmeanSquareLoss2 # need for source code execution


# BinaryEntropy loss:
FBinaryEntropyLoss(x::Array{Float64,2},y::Array{Float64,2})=begin aux=prod(size(x)); return (sum(x.*log(x./y)+(1.-x).*log((1.-x)./(1.-y)))/aux,aux); end
DBinaryEntropyLoss=Array(Function,2)
DBinaryEntropyLoss[1]=dBinaryEntropyLoss1(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Array{Float64,2})=t*log(x.*(1.-y)./(y.*(1.-x)))/aux
DBinaryEntropyLoss[2]=dBinaryEntropyLoss2(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Array{Float64,2})=t*((x-y)./(y.*(y-1.)))/aux
DBinaryEntropyLoss[1]=dBinaryEntropyLoss1(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Float64)=t*log(x.*(1-y)./(y.*(1.-x)))/aux
DBinaryEntropyLoss[2]=dBinaryEntropyLoss2(x::Array{Float64,2},y::Array{Float64,2},self,aux,t::Float64)=t*((x-y)./(y.*(y-1.)))/aux
Derivative[FBinaryEntropyLoss]=DBinaryEntropyLoss
export FBinaryEntropyLoss
export dBinaryEntropyLoss1,dBinaryEntropyLoss2 # need for source code execution


#softmax function and derivative
Fsoftmax(x::Array{Float64,2})=(exp(x)./sum(exp(x),1),[])
Dsoftmax=Array(Function,1)
Dsoftmax[1]=Dsoftmax1(x::Array{Float64,2},self,aux,t::Array{Float64,2})=self.*(t-repmat(sum(self.*t,1),size(self,1),1))
Derivative[Fsoftmax]=Dsoftmax
export Fsoftmax,Dsoftmax
export Dsoftmax1 # need for source code execution

#Useful to code KL(p,softmax(x)) function since this has a computationally convenient derivative:
FKLsoftmax(p::Array{Float64,2},x::Array{Float64,2})=begin DN=prod(size(x));Z=sum(exp(x),1);aux=(Z,DN); return ((sum(p.*(log(p)-x))+sum(log(Z)))/DN,aux); end
DKLsoftmax=Array(Function,2)
DKLsoftmax[1]=DKLsoftmax1(p::Array{Float64,2},x::Array{Float64,2},self,aux,t)=t.*(1+log(p)-x) # CHECK!
DKLsoftmax[2]=DKLsoftmax2(p::Array{Float64,2},x::Array{Float64,2},self,aux,t)=t.*(exp(x)./aux[1]-p)./aux[2]
Derivative[FKLsoftmax]=DKLsoftmax
export FKLsoftmax,DKLsoftmax
export DKLsoftmax1,DKLsoftmax2 # need for source code execution


# KL Loss
FKLLoss(p::Array{Float64,2},q::Array{Float64,2})=begin DN=prod(size(x));(sum(p.*(log(p)-log(q)))/DN,DN); end
DKLLoss=Array(Function,2)
DKLLoss[1]=DKLLoss1(p::Array{Float64,2},q::Array{Float64,2},self,aux,t)=t.*(1+log(p)-log(q))/aux
DKLLoss[2]=DKLLoss2(p::Array{Float64,2},q::Array{Float64,2},self,aux,t)=t.*(p./q)./aux
Derivative[FKLLoss]=DKLLoss
export FKLLoss,DKLLoss
export DKLLoss1,DKLLoss2 # need for source code execution

# Multinomial Logistic Loss
FMultLogisticLoss(c::BitArray{2},x::Array{Float64,2})=begin DN=prod(size(x));Z=sum(exp(x),1);aux=(Z,DN); return ((-sum(x[c])+sum(log(Z)))/DN,aux); end
DMultLogisticLoss=Array(Function,2)
DMultLogisticLoss[1]=DMultLogisticLoss1(c::BitArray{2},x::Array{Float64,2},self,aux,t)=nan # not needed
DMultLogisticLoss[2]=DMultLogisticLoss2(c::BitArray{2},x::Array{Float64,2},self,aux,t)=begin p=zeros(size(c)); p[c]=1.; return t.*(exp(x)./aux[1]-p)./aux[2]; end
Derivative[FMultLogisticLoss]=DMultLogisticLoss
export FMultLogisticLoss,DMultLogisticLoss
export DMultLogisticLoss1,DMultLogisticLoss2 # need for source code execution


#Logistic Loss -sum(log(sigma(c.*x)))/prod(size(x)) c[i] is +1 or -1 class variable
FLogisticLoss(c::Array{Float64,2},x::Array{Float64,2})=begin DN=prod(size(x)); aux=(1./(1.+exp(-c.*x)),DN); return (-sum(log(aux[1]))/DN,aux); end
DLogisticLoss=Array(Function,2)
DLogisticLoss[1]=DLogisticLoss1(c::Array{Float64,2},x::Array{Float64,2},self,aux,t)=-t.*(1.-aux[1]).*x/aux[2]
DLogisticLoss[2]=DLogisticLoss2(c::Array{Float64,2},x::Array{Float64,2},self,aux,t)=-t.*(1.-aux[1]).*c/aux[2]

FLogisticLoss(c::Array{Float64},x::Array{Float64})=begin DN=prod(size(x)); aux=(1./(1.+exp(-c.*x)),DN); return (-sum(log(aux[1]))/DN,aux); end
DLogisticLoss=Array(Function,2)
DLogisticLoss[1]=DLogisticLoss1(c::Array{Float64,1},x::Array{Float64,2},self,aux,t)=-t.*(1.-aux[1]).*x/aux[2]
DLogisticLoss[2]=DLogisticLoss2(c::Array{Float64,1},x::Array{Float64,2},self,aux,t)=-t.*(1.-aux[1]).*c/aux[2]

Derivative[FLogisticLoss]=DLogisticLoss
export FLogisticLoss,DLogisticLoss
export DLogisticLoss1,DLogisticLoss2 # need for source code execution

export Derivative
