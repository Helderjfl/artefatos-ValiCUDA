#!/bin/bash
programa="count_ocurrence"
criterios=( "todos-nos" "todos-nos-grid" "todos-nos-host" "todos-nos-sinc" "todos-nos-sinc-host" "todos-nos-sinc-grid"
            "todas-arestas" "todas-arestas-grid" "todas-arestas-host" "todas-arestas-sinc"
            "todos-c-usos" "todos-c-usos-grid" "todos-c-usos-host" 
            "todos-p-usos" "todos-p-usos-grid" "todos-p-usos-host"
            "todos-usos" "todos-usos-grid" "todos-usos-host"
            "todos-s-c-usos" "todos-s-p-usos" "todos-s-usos"
            "todos-bloco-c-usos-grid" "todos-bloco-p-usos-grid"
            "todos-global-c-usos-grid" "todos-global-p-usos-grid"
            )

if [[ $1 = "-inst" || $2 = "-inst" || $3 = "-inst" || $4 = "-inst" ]]; then
    clava ~/Github/ClavaCUDA/src/instrumentCUDA.lara -i ~/Github/ClavaCUDA --verbose 0 -p ${programa}.cu
fi

cp input*.txt Tests/${programa}.cu/
cd Tests/${programa}.cu

if [[ $1 = "-elem" || $2 = "-elem" || $3 = "-elem" || $4 = "-elem" ]]; then
    vali_elem 5 "Host(0)" "Grid0(1)" -cuda
fi


if [[ $1 = "-run" || $2 = "-run" || $3 = "-run" || $4 = "-run" ]]; then
    nvcc ${programa}.cu -o ${programa}.o

    # vali_exec -cuda 0 run 2 ${programa}.o "1 1" input0.txt
    # vali_exec -cuda 1 run 2 ${programa}.o "2048 1" input1.txt
    vali_exec -cuda 0 run 2 ${programa}.o "8192 1" input1.txt
    vali_exec -cuda 1 run 2 ${programa}.o "8192 2" input1.txt
    # vali_exec -cuda 4 run 2 ${programa}.o "8192 9" input2.txt

    # for i in {1..3};
    # do
    #     vali_exec -cuda $(($i-1)) run 2 ${programa}.o ${i} < "input$i.txt"
    # done
fi


if [[ $1 = "-eval" || $2 = "-eval" || $3 = "-eval" || $4 = "-eval" ]]; then
    rm -r coverage/CSV
    # vali_eval todos-nos 2 "Host(0)" "Grid(1)" -cuda 4096

    for criterio in ${criterios[@]};
    do 
        echo " "
        echo "---------------------------------------------------------------------------------"
        echo "Criterio: ${criterio}"
        vali_eval "${criterio}" 2 "Host(0)" "Grid(1)" -cuda 8192
        echo " "
    done
fi