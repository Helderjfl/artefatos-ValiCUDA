programa="matrix_mult"
clava ~/Github/ClavaCUDA/src/instrumentCUDA.lara -i ~/Github/ClavaCUDA --verbose 0 -p ${programa}.cu
cp input*.txt Tests/${programa}.cu/
cd Tests/${programa}.cu
# vali_elem 2 "Host(0)" "Grid0(1)" -cuda
# nvcc ${programa}.cu -o ${programa}.o
# vali_exec -cuda 0 run 2 ${programa}.o "4 16" < input1.txt
# vali_exec -cuda 1 run 2 ${programa}.o "32 16" < input2.txt
# vali_exec -cuda 2 run 2 ${programa}.o "64 16" < input3.txt

rm -r coverage/CSV

# vali_eval todos-nos 2 "Host(0)" "Grid(1)" -cuda 256
criterios=( "todos-nos" "todos-nos-grid" "todos-nos-host" "todos-nos-sinc" "todos-nos-sinc-host" "todos-nos-sinc-grid"
            "todas-arestas" "todas-arestas-grid" "todas-arestas-host" "todas-arestas-sinc"
            "todos-c-usos" "todos-c-usos-grid" "todos-c-usos-host" 
            "todos-p-usos" "todos-p-usos-grid" "todos-p-usos-host"
            "todos-usos" "todos-usos-grid" "todos-usos-host"
            "todos-s-c-usos" "todos-s-p-usos" "todos-s-usos"
            "todos-bloco-c-usos-grid" "todos-bloco-p-usos-grid"
            "todos-global-c-usos-grid" "todos-global-p-usos-grid")
# for j in {0..25};
# do 
#     echo " "
#     echo "---------------------------------------------------------------------------------"
# 	echo "Criterio: ${criterios[$j]}"
#     vali_eval "${criterios[$j]}" 2 "Host(0)" "Grid(1)" -cuda 4096
#     echo " "
# done

cd ../..