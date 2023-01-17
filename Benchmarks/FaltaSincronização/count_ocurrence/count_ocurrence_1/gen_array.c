#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[])
{
    int N = atoi(argv[1]);
    FILE *fp;
    char arquivo[50] = "input";

    strcat(arquivo, argv[2]);
    strcat(arquivo, ".txt");

    fp = fopen(arquivo, "w");

    int cont = 1;

    cont = 2048;
    for(int i = 0; i < N; i++){
        fprintf(fp, "%d ", rand() % 10);
    }

    fclose(fp);

    return 0;
}
