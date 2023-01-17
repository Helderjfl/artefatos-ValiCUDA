for j in {1..32};
do 
    # ./gen_matrix.o $j $(($j+32))
    ./gen_matrix.o $j $j
done