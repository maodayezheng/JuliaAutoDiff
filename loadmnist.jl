function loadmnist()
    alldata=matread("mnist-original.mat")
    images=(float(alldata["data"]))./255
    return images, alldata["label"]
end
