using Distributed
using JLD2, FileIO
using LinearAlgebra;
using Random;
using ClusterManagers

ns=[16,32,64,128,256,512]
p_singles=collect(0.0:0.01:0.4)
p_unitaries=[0.3]
runNum=1000

##---------------------------------------
function checkforFile(dir,fileName)
    for s in readdir("../")
        if s==fileName
            return true
        end
    end
    return false
end

function lockDir(dir="./")
    attempts=0
    while checkforFile(dir,"lock")
       attempts+=1
        println("attempt $attempts failed to lock the dir")
       if attempts==60
                println("failed to lock the directory, exiting now ...")
                exit()
       end
       sleep(5)
    end
    close(open(string(dir,"lock"),"w"))
    return
end

function unlockDir(dir="./")
    cmd=`rm $(dir)lock`
    run(cmd)
end
function getSeqId(prefix,delimiter='.')
    seqId=0
    len=length(prefix)
    for fname in readdir()
        if startswith(fname,prefix) && findlast(isequal(delimiter),fname)!==nothing
            serial=fname[len+1:findlast(isequal(delimiter),fname)-1]
            if tryparse(Int64,serial)!==nothing
                seqId=max(parse(Int64,serial),seqId)
            end
        end
    end
    return seqId+1
end
##---------------------------------------

sleep(10*rand())

#start worker processes 
lockDir("../")
addprocs(SlurmManager(parse(Int64,ARGS[2])))
unlockDir("../")

@everywhere include("GKToolboxv3.5.jl");
@everywhere using .GKToolboxv3
@everywhere using LinearAlgebra;
@everywhere using Random;



validPairsNum=0
for p in p_unitaries
    for q in p_singles
        if p+q <= 1.0
            global validPairsNum+=1
        end
    end
end
println("Valid Pairs #", validPairsNum)



#race conditions could happen in general
if endswith(@__FILE__,".jl") && length(ARGS)!=0
    #used for passing the file name as the argument
    filename=string("data",ARGS[1],".jld2")
else
    #when the code is run on jupyter or no filename is passed
    filename=string("data",getSeqId("data"),".jld2")
end

#it will build the file
close(jldopen(filename, "w"))
println("filename:",filename)

jldopen(filename,"a+") do myfile
    myfile["pars/ns"]=ns
    myfile["pars/p_unitaries"]=p_unitaries
    myfile["pars/p_singles"]=p_singles
    myfile["pars/runNum"]=runNum
end


#compute each process's share
n1=runNum%nworkers()
ensembleSize=[ones(Int,n1)*Int(floor(runNum/nworkers())+1);ones(Int,nworkers()-n1)*Int(floor(runNum/nworkers()))]

resultsChannel= RemoteChannel(()->Channel{EntanglementData}(10*nworkers()));
gatherdData=Array{EntanglementData,3}(undef, length(ns),length(p_unitaries),length(p_singles))


#distribute jobs
for i in eachindex(workers())
    remote_do(mineTheDataU_Ancilla_v2, workers()[i], ns,p_unitaries,p_singles, ensembleSize[i], resultsChannel)
end

#gather the results and write it on the disk
@time for _ in 1:validPairsNum*length(ns)*nworkers()


    rd=take!(resultsChannel)
    println("data recieved for n=",ns[rd.n_id], ", p_unitary=",p_unitaries[rd.p_unitary_id], "and p_single=",p_singles[rd.p_single_id]," for ensembleSize of ", rd.ensembleSize, " computed in ",rd.timeConsumed )

    n_id=rd.n_id
    p_unitary_id=rd.p_unitary_id
    p_single_id=rd.p_single_id

    if !isassigned(gatherdData,n_id,p_unitary_id,p_single_id)

        gatherdData[n_id,p_unitary_id,p_single_id]=rd
    else
        gd=gatherdData[n_id,p_unitary_id,p_single_id]

        gd.s_topo+=rd.s_topo
        gd.quarterLengthEnt+=rd.quarterLengthEnt
        gd.halfLengthEnt+=rd.halfLengthEnt
        gd.finalEntProf+=rd.finalEntProf
        gd.ensembleSize+=rd.ensembleSize
        gd.ancillaEntropy+=rd.ancillaEntropy
    end


    if gatherdData[n_id,p_unitary_id,p_single_id].ensembleSize==runNum

        gd=gatherdData[n_id,p_unitary_id,p_single_id]

        gd.s_topo/=runNum
        gd.quarterLengthEnt/=runNum
        gd.halfLengthEnt/=runNum
        gd.finalEntProf/=runNum
        gd.ancillaEntropy/=runNum


        jldopen(filename, "a+") do file
            file[string("data/",ns[n_id],'_',p_unitaries[p_unitary_id],'_',p_singles[p_single_id])] = gd
        end
    end

end
