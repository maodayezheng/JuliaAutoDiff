function loadmnist()
    #using Winston# for plotting
    #using MAT# work with matlab files
    alldata=matread("../../../../data/mnist-original.mat")
    images=(float(alldata["data"]))./255
    #spy(reshape(images[:,1],28,28)')
    #colormap("grays")
    #imagesc(reshape(images[:,1],28,28)',(0,1))
    return images, alldata["label"]
end
