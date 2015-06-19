function gradcheck(value,net,showgrad=false)
    # requires Cartesian package
    println("gradcheck...")
    v=deepcopy(value) # so we don't modify the calling argument
    (v,g)=ADeval!(v,net)
    epsilon=1e-8; tol=1e-6
    gemp=deepcopy(v)
    for par=1:length(net.node) # loop over the nodes in the network
        if net.node[par].returnderivative
            @forcartesian i size(v[par]) begin
                vold=v[par][i...]
                v[par][i...]=vold+epsilon
                (fplus,dummy)=ADeval!(v,net;returnall=false)
                v[par][i...]=vold-epsilon
                (fminus,dummy)=ADeval!(v,net;returnall=false)
                gemp[par][i...]=0.5*(fplus-fminus)/epsilon
                v[par][i...]=vold
            end
            diff=mean(abs(g[par]-gemp[par]))
            reldiff=mean(abs((g[par]-gemp[par])./(realmin()+g[par])))
            println("node $par: absolute difference between analytic and empirical gradient=$diff")
            println("node $par: relative difference between analytic and empirical gradient=$reldiff")
            if diff>tol
                if showgrad
                println("analytic gradient:"); println(g[par])
                println("empirical gradient:"); println(gemp[par])
                end
                println("failed: analytic and empiricial gradient mismatch more than $tol")
            else
                println("passed")
            end
        end
    end
end
