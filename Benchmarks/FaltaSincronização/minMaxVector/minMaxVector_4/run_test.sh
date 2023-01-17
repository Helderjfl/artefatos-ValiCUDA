#!/bin/bash
programa="minMaxVector"
criterios=( "todos-nos" "todos-nos-grid" "todos-nos-host" "todos-nos-sinc" "todos-nos-sinc-host" "todos-nos-sinc-grid"
            "todas-arestas" "todas-arestas-grid" "todas-arestas-host" "todas-arestas-sinc"
            "todos-c-usos" "todos-c-usos-grid" "todos-c-usos-host" 
            "todos-p-usos" "todos-p-usos-grid" "todos-p-usos-host"
            "todos-usos" "todos-usos-grid" "todos-usos-host"
            "todos-s-c-usos" "todos-s-p-usos" "todos-s-usos"
            "todos-bloco-c-usos-grid" "todos-bloco-p-usos-grid"
            "todos-global-c-usos-grid" "todos-global-p-usos-grid")

if [[ $1 = "-inst" || $2 = "-inst" || $3 = "-inst" || $4 = "-inst" ]]; then
    clava ~/Github/ClavaCUDA/src/instrumentCUDA.lara -i ~/Github/ClavaCUDA --verbose 0 -p ${programa}.cu
fi

cp input*.txt Tests/${programa}.cu/
cd Tests/${programa}.cu

if [[ $1 = "-elem" || $2 = "-elem" || $3 = "-elem" || $4 = "-elem" ]]; then
    vali_elem 5 "Host(0)" "Grid0(1)" "Grid1(2)" "Grid2(3)" "Grid3(4)" -cuda
fi


if [[ $1 = "-run" || $2 = "-run" || $3 = "-run" || $4 = "-run" ]]; then
    nvcc ${programa}.cu -o ${programa}.o

    vali_exec -cuda 0 run 2 ${programa}.o "2048" input1.txt
    vali_exec -cuda 1 run 2 ${programa}.o "2048" input2.txt
    vali_exec -cuda 2 run 2 ${programa}.o "1" input3.txt
    vali_exec -cuda 3 run 2 ${programa}.o "1025" input7.txt
    # vali_exec -cuda 3 run 2 ${programa}.o "257" input4.txt
    # vali_exec -cuda 4 run 2 ${programa}.o "513" input5.txt
    # vali_exec -cuda 5 run 2 ${programa}.o "769" input6.txt
    # vali_exec -cuda 6 run 2 ${programa}.o "1025" input7.txt
    # vali_exec -cuda 7 run 2 ${programa}.o "1281" input8.txt
    # vali_exec -cuda 8 run 2 ${programa}.o "1537" input9.txt
    # vali_exec -cuda 9 run 2 ${programa}.o "1793" input10.txt
    # vali_exec -cuda 2 run 2 ${programa}.o "8" input3.txt

    # 1~2
    # vali_exec -cuda 0 run 2 ${programa}.o "2048" input1.txt
    # vali_exec -cuda 1 run 2 ${programa}.o "2048" input2.txt

    # 1~3
    # vali_exec -cuda 0 run 2 ${programa}.o "2048" input1.txt
    # vali_exec -cuda 1 run 2 ${programa}.o "2048" input2.txt
    # vali_exec -cuda 2 run 2 ${programa}.o "1" input3.txt

    # 1~10
    # vali_exec -cuda 0 run 2 ${programa}.o "2048" input1.txt
    # vali_exec -cuda 1 run 2 ${programa}.o "2048" input2.txt
    # vali_exec -cuda 2 run 2 ${programa}.o "1" input3.txt
    # vali_exec -cuda 3 run 2 ${programa}.o "257" input4.txt
    # vali_exec -cuda 4 run 2 ${programa}.o "513" input5.txt
    # vali_exec -cuda 5 run 2 ${programa}.o "769" input6.txt
    # vali_exec -cuda 6 run 2 ${programa}.o "1025" input7.txt
    # vali_exec -cuda 7 run 2 ${programa}.o "1281" input8.txt
    # vali_exec -cuda 8 run 2 ${programa}.o "1537" input9.txt
    # vali_exec -cuda 9 run 2 ${programa}.o "1793" input10.txt

    # 1~4
    # vali_exec -cuda 0 run 2 ${programa}.o "2048" input1.txt
    # vali_exec -cuda 1 run 2 ${programa}.o "2048" input2.txt
    # vali_exec -cuda 2 run 2 ${programa}.o "1" input3.txt
    # vali_exec -cuda 3 run 2 ${programa}.o "320" input11.txt

    # for i in {1..3};
    # do
    #     vali_exec -cuda $(($i-1)) run 2 ${programa}.o ${i} < "input$i.txt"
    # done
fi


if [[ $1 = "-eval" || $2 = "-eval" || $3 = "-eval" || $4 = "-eval" ]]; then
    rm -r coverage/CSV
    # vali_eval todos-nos 2 "Host(0)" "Grid(1)" -cuda 4096

    for j in {0..25};
    do 
        echo " "
        echo "---------------------------------------------------------------------------------"
        echo "Criterio: ${criterios[$j]}"
        vali_eval "${criterios[$j]}" 5 "Host(0)" "Grid(1)" "Grid(2)" "Grid(3)" "Grid(4)" -cuda 2048 256 2048 256
        echo " "
    done
fi