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

    // for(int i = 0; i < N; i++){
    //     for(int j = 0; j < N; j++)
    //         fprintf(fp, "%d ", 0);
    //     fprintf(fp, "\n");
    // }

    

    cont = 1;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            fprintf(fp, "%d ", 2);
        fprintf(fp, "\n");
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            if(j+i == N-1)
                fprintf(fp, "%d ", 1);
            else
                fprintf(fp, "%d ", 0);
        fprintf(fp, "\n");
    }


    fclose(fp);

    return 0;
}
