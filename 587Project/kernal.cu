
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace std;

__constant__ int lookup_table[256];

__global__ void generate_candidate(uint32_t *itemSets, uint32_t *tranLists, 
                                   int *itemSet_length, int *itemSets_size, 
                                   int *itemSets_d2, int *tranLists_d2,
                                   uint32_t *n_itemSets, uint32_t *n_tranLists, 
                                   int *next_itemSets_size) {
    int num_of_thread = gridDim.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int itemSets_per_thread = ceil(((double) *itemSets_size) / num_of_thread);    

    // Initialize all outout arrays to 0
    for (int i = 0; i < (*next_itemSets_size) * (*itemSets_d2); i++) {
        n_itemSets[i] = 0;
    }
    for (int i = 0; i < (*next_itemSets_size) * (*tranLists_d2); i++) {
        n_tranLists[i] = 0;
    }    

    for (int i = idx; i < (*itemSets_size) - 1; i += num_of_thread) {
        for (int j = i + 1; j < *itemSets_size; j++) {
            bool is_joinable = true;
            
            // Check bits before (itemSet_length - 1)th "1", join if equal
            int ones_to_check = (*itemSet_length - 1);
            for (int k = 0; k < *itemSets_d2; k++) {
                if (ones_to_check == 0) {
                    break;
                }

                uint32_t left = itemSets[k + i * (*itemSets_d2)];
                uint32_t right = itemSets[k + j * (*itemSets_d2)];
                int sum_left = 0;
                int sum_right = 0;
                uint32_t mask = 255;
                uint32_t part_left, part_right;

                for (int l = 0; l < 4; l++) {
                    part_left = (itemSets[k + i * (*itemSets_d2)] >> (8 * l)) & mask;
                    sum_left += lookup_table[part_left];
                    part_right = (itemSets[k + j * (*itemSets_d2)] >> (8 * l)) & mask;
                    sum_right += lookup_table[part_right];
                }
                
                //printf("ones to check:%d, sum_left:%d, sum_right:%d\n", ones_to_check, sum_left, sum_right);
                if (ones_to_check <= sum_left && ones_to_check <= sum_right) {
                    if (sum_left == sum_right && ones_to_check == sum_left) {
                        if (left != right) {
                            is_joinable = false;
                        }
                        break;
                    } else {
                        // Check bit by bit until enough "1" met
                        mask = 1;
                        for (int l = 0; l < 32; l++) {
                            uint32_t masked_left = left & mask;
                            uint32_t masked_right = right & mask;
                            if (masked_left != masked_right) {
                                is_joinable = false;
                                break;
                            } else {
                                if (masked_left == mask) {
                                    ones_to_check--;
                                    if (ones_to_check == 0) {
                                        break;
                                    }
                                }
                            }
                            mask = mask << 1;
                        }
                        break;
                    }
                } else if (sum_left != sum_right) {
                    is_joinable = false;
                    break;
                } else {
                    if (left != right) {
                        is_joinable = false;
                        break;
                    }
                    ones_to_check -= sum_left;
                }
            }

            /* Output valid joined itemSets, sum from start to end and j - (i + 1) 
               to get output position */
            if (is_joinable) {
                int output_pos;
                if (i == 0) {
                    output_pos = j - (i + 1);
                } else {
                    int start = *itemSets_size - 1;
                    int end = start - (i - 1);
                    output_pos = (start + end) * i / 2 + (j - (i + 1));
                }
                
                //printf("i:%d, j:%d, output_pos:%d, is joinable:%d\n",i ,j , output_pos, is_joinable);
                for (int k = 0; k < *itemSets_d2; k++) {
                    n_itemSets[k + output_pos * (*itemSets_d2)] = 
                        itemSets[k + i * (*itemSets_d2)] | itemSets[k + j * (*itemSets_d2)];
                }
                for (int k = 0; k < *tranLists_d2; k++) {
                    n_tranLists[k + output_pos * (*tranLists_d2)] = 
                        tranLists[k + i * (*tranLists_d2)] & tranLists[k + j * (*tranLists_d2)];
                }
            } else {
                break;
            }
        }
    }
}

__global__ void count_tranx(uint32_t *tranLists, int *itemSets_size, int *tranLists_d2, 
                            int *countLists) {
    int num_of_thread = gridDim.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int itemSets_per_thread = ceil(((double) *itemSets_size) / num_of_thread);

    // Initialize output array to 0
    for (int i = 0; i < (*itemSets_size) * (*tranLists_d2); i++) {
        countLists[i] = 0;
    }
    
    for (int i = idx; i < *itemSets_size; i += num_of_thread) {
        for (int j = 0; j < *tranLists_d2; j++) {
            int sum = 0;
            uint32_t mask = 255;
            uint32_t part;
            for (int k = 0; k < 4; k++) {
                part = (tranLists[j + i * (*tranLists_d2)] >> (8 * k)) & mask;
                sum += lookup_table[part];
            }
            //printf("i:%d, sum:%d\n", i, sum);
            countLists[j + i * (*tranLists_d2)] = sum;
        }
    }
}

int main()
{
    float min_sup = 0.005f;
    int num_of_block = 1;
    int thread_per_block = 4;
    int itemSet_length = 1;
    
    // Read file to generate initial itemSets and tranLists
    map<int, vector<int>> item_to_tranxs;
    ifstream ifs("1000.dat");
    string line;
    int tranxID = 0;
    while (getline(ifs, line)) {
        istringstream iss(line);
        while (iss) {
            int itemID;
            iss >> itemID;
            //itemID--;
            if (item_to_tranxs.find(itemID) != item_to_tranxs.end()) {
                item_to_tranxs[itemID].push_back(tranxID);
            } else {
                item_to_tranxs[itemID] = vector<int>();
                item_to_tranxs[itemID].push_back(tranxID);
            }
        }
        tranxID++;
    }
    
    int num_of_item = item_to_tranxs.size();
    int num_of_tranx = tranxID;
    int itemSets_size = item_to_tranxs.size();
    int itemSets_d2 = ceil(((float) num_of_item) / 32);
    int tranLists_d2 = ceil(((float) num_of_tranx) / 32);
    uint32_t *itemSets = new uint32_t[itemSets_size * itemSets_d2];
    uint32_t *tranLists = new uint32_t[itemSets_size * tranLists_d2];

    for (int i = 0; i < itemSets_size; i++) {        
        for (int j = 0; j < itemSets_d2; j++) {
            itemSets[j + i * itemSets_d2] = 0;
        }
    }
    for (int i = 0; i < item_to_tranxs.size(); i++) {
        int idx_in_d2 = i / 32;
        uint32_t value = pow(2.0, i % 32);
        itemSets[idx_in_d2 + i * itemSets_d2] = value;
    }
    
    for (int i = 0; i < item_to_tranxs.size(); i++) {
        vector<uint32_t> values(tranLists_d2, 0);
        vector<int> tranxIDs = item_to_tranxs[i];
        for (int j = 0; j < tranxIDs.size(); j++) {
            int idx_in_d2 = tranxIDs[j] / 32;
            uint32_t value = pow(2.0, tranxIDs[j] % 32);
            values[idx_in_d2] += value;
        }
        for (int j = 0; j < tranLists_d2; j++) {
            tranLists[j + i * tranLists_d2] = values[j];
        }        
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // Build lookup table, and copy it to GPU
    int host_lookup_table[256];
    for (int i = 0; i < 256; i++) {
        int count = 0;
        unsigned int num = i;
        while (num != 0) {
            if (num & 1 == 1) {
                count++;
            }
            num = num >> 1;
        }
        host_lookup_table[i] = count;
    }
    cudaMemcpyToSymbol(lookup_table, host_lookup_table, 256 * sizeof(int));

    // Set 2nd dimension of itemSets and tranLists, they should be fixed
    int *dev_itemSets_d2 = 0;
    int *dev_tranLists_d2 = 0;
    cudaMalloc((void**)&dev_itemSets_d2, sizeof(int));
    cudaMalloc((void**)&dev_tranLists_d2, sizeof(int));
    cudaMemcpy(dev_itemSets_d2, &itemSets_d2, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_tranLists_d2, &tranLists_d2, sizeof(int), cudaMemcpyHostToDevice);

    clock_t t = clock();
    while (itemSets_size != 0 && itemSet_length <= num_of_item - 1) {
        // generate candidate itemSets
        int next_itemSets_size = itemSets_size * (itemSets_size - 1) / 2;
        uint32_t *next_itemSets = new uint32_t[next_itemSets_size * itemSets_d2];
        uint32_t *next_tranLists = new uint32_t[next_itemSets_size * tranLists_d2];
        
        uint32_t *dev_itemSets = 0;
        uint32_t *dev_tranLists = 0;
        int *dev_itemSets_size = 0; 
        int *dev_itemSet_length = 0;
        uint32_t *dev_next_itemSets = 0;
        uint32_t *dev_next_tranLists = 0;
        int *dev_next_itemSets_size = 0;
    
        cudaMalloc((void**)&dev_itemSets, itemSets_size * itemSets_d2 * sizeof(uint32_t));
        cudaMalloc((void**)&dev_tranLists, itemSets_size * tranLists_d2 * sizeof(uint32_t));
        cudaMalloc((void**)&dev_itemSets_size, sizeof(int));
        cudaMalloc((void**)&dev_itemSet_length, sizeof(int));
        cudaMalloc((void**)&dev_next_itemSets, next_itemSets_size * itemSets_d2 * 
            sizeof(uint32_t));
        cudaMalloc((void**)&dev_next_tranLists, next_itemSets_size * tranLists_d2 * 
            sizeof(uint32_t));
        cudaMalloc((void**)&dev_next_itemSets_size, sizeof(int));

        cudaMemcpy(dev_itemSets, itemSets, itemSets_size * itemSets_d2 * sizeof(uint32_t), 
            cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tranLists, tranLists, itemSets_size * tranLists_d2 * sizeof(uint32_t), 
            cudaMemcpyHostToDevice);
        cudaMemcpy(dev_itemSets_size, &itemSets_size, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_itemSet_length, &itemSet_length, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_next_itemSets_size, &next_itemSets_size, sizeof(int), 
            cudaMemcpyHostToDevice);
        
        clock_t t1 = clock();
        generate_candidate<<<num_of_block, thread_per_block>>>(dev_itemSets, dev_tranLists, 
            dev_itemSet_length, dev_itemSets_size, dev_itemSets_d2, dev_tranLists_d2, 
            dev_next_itemSets, dev_next_tranLists, dev_next_itemSets_size);

        cudaDeviceSynchronize();
        t1 = clock() - t1;
        cout << "Time for candidate generation:" << ((float)t1) / CLOCKS_PER_SEC << endl;

        // Copy results back
        cudaMemcpy(next_itemSets, dev_next_itemSets, next_itemSets_size * itemSets_d2 * 
            sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(next_tranLists, dev_next_tranLists, next_itemSets_size * tranLists_d2 *
            sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Pick valid joined ItenSets 
        uint32_t *joined_itemSets = new uint32_t[next_itemSets_size * itemSets_d2];
        uint32_t *joined_tranLists = new uint32_t[next_itemSets_size * tranLists_d2];
        int cursor = 0;
        for (int i = 0; i < next_itemSets_size; i++) {
            for (int j = 0; j < itemSets_d2; j++) {
                if (next_itemSets[j + i * itemSets_d2] != 0) {
                    for (int k = 0; k < itemSets_d2; k++) {
                        joined_itemSets[k + cursor * itemSets_d2] = 
                            next_itemSets[k + i * itemSets_d2];
                    }
                    for (int k = 0; k < tranLists_d2; k++) {
                        joined_tranLists[k + cursor * tranLists_d2] = 
                            next_tranLists[k + i * tranLists_d2];
                    }
                    cursor++;
                    break;
                }
            }
        }
        
        if (cursor == 0) {
            cout << "No joined ItemSets found!" << endl;
            break;
        }

        // Count transactions
        int joined_itemSets_size = cursor;
        int *countLists = new int[joined_itemSets_size * tranLists_d2];

        uint32_t *dev_joined_tranLists = 0;
        int *dev_joined_itemSets_size = 0;
        int *dev_countLists = 0;

        cudaMalloc((void**)&dev_joined_tranLists, joined_itemSets_size * tranLists_d2 *
            sizeof(uint32_t));
        cudaMalloc((void**)&dev_joined_itemSets_size, sizeof(int));
        cudaMalloc((void**)&dev_countLists, joined_itemSets_size * tranLists_d2 *
            sizeof(int));

        cudaMemcpy(dev_joined_tranLists, joined_tranLists, joined_itemSets_size * 
            tranLists_d2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_joined_itemSets_size, &joined_itemSets_size, sizeof(int), 
            cudaMemcpyHostToDevice);

        t1 = clock();
        count_tranx<<<num_of_block, thread_per_block>>>(dev_joined_tranLists, 
            dev_joined_itemSets_size, dev_tranLists_d2, dev_countLists);

        cudaDeviceSynchronize();
        t1 = clock() - t1;
        cout << "Time for counting:" << ((float)t1) / CLOCKS_PER_SEC << endl;

        // Copy results back
        cudaMemcpy(countLists, dev_countLists, joined_itemSets_size * tranLists_d2 * 
            sizeof(int), cudaMemcpyDeviceToHost);
        
        // Pick frequent ItemSets
        delete[] itemSets;
        delete[] tranLists;
        itemSets = new uint32_t[joined_itemSets_size * itemSets_d2];
        tranLists = new uint32_t[joined_itemSets_size * tranLists_d2];

        cursor = 0;
        for (int i = 0; i < joined_itemSets_size; i++) {
            int count = 0;
            for (int j = 0; j < tranLists_d2; j++) {
                count += countLists[j + i * tranLists_d2];
            }
            float support = ((float) count) / num_of_tranx;
            if (support >= min_sup) {
                for (int j = 0; j < itemSets_d2; j++) {
                    itemSets[j + cursor * itemSets_d2] = 
                        joined_itemSets[j + i * itemSets_d2];
                }
                for (int j = 0; j < tranLists_d2; j++) {
                    tranLists[j + cursor * tranLists_d2] = 
                        joined_tranLists[j + i * tranLists_d2];
                }
                cursor++;
            }
        }
        itemSets_size = cursor;
        itemSet_length++;

        // Output frequent itemSet
        cout << "frequent itemSets_size: " << itemSets_size << endl;        
        /* for (int i = 0; i < itemSets_size; i++) {
            for (int j = 0; j < itemSets_d2; j++) {
                cout << itemSets[j + i * itemSets_d2] << " ";
            }
        }
        cout << endl; */

        // Clean up
        cudaFree(dev_itemSets);
        cudaFree(dev_tranLists);
        cudaFree(dev_itemSets_size);
        cudaFree(dev_itemSet_length);
        cudaFree(dev_next_itemSets);
        cudaFree(dev_next_tranLists);
        cudaFree(dev_next_itemSets_size);
        cudaFree(dev_joined_tranLists);
        cudaFree(dev_joined_itemSets_size);
        cudaFree(dev_countLists);
        delete[] next_itemSets;
        delete[] next_tranLists;
        delete[] joined_itemSets;
        delete[] joined_tranLists;
        delete[] countLists;
    }
    t = clock() - t;
    cout << "Time:" << ((float)t) / CLOCKS_PER_SEC << endl;
    return 0;
}
