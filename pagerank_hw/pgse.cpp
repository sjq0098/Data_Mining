#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <queue>
#include <tuple>
#include <algorithm>
#include <filesystem>
using namespace std;

namespace fs = std::filesystem;

// —— 全局配置 —— 
const double damping_factor       = 0.85;     // 跳转概率
const double epsilon    = 1e-6;     // 收敛阈值
const int    max_iter   = 100;      // 最大迭代次数
const int    block_size = 1000;     // 每个条带大小
const std::string data_file = "Data.txt";

// 读取边文件，统计 outdeg 并分条带 —— 
void create_edge_stripes(int N, int num_stripes) {
    // 打开所有条带输出流
    vector<ofstream> stripe_fs;
    stripe_fs.reserve(num_stripes);
    for(int i = 0; i < num_stripes; ++i) {
        stripe_fs.emplace_back("stripe_" + std::to_string(i) + ".txt");
    }
    // 读入边，写到对应条带
    ifstream fin(data_file);
    int u, v;
    while(fin >> u >> v) {
        int sid = v / block_size;
        stripe_fs[sid] << u << " " << v << "\n";
    }
    for(auto &ofs : stripe_fs) ofs.close();
}

// 初始化 PageRank 向量，按条带写磁盘 
void write_initial_r(int N, int num_stripes) {
    for(int sid = 0; sid < num_stripes; ++sid) {
        int start = sid * block_size;
        int end   = std::min((sid+1)*block_size, N);
        ofstream rf("r_stripe_" + std::to_string(sid) + ".txt");
        for(int i = start; i < end; ++i) {
            rf << (1.0 / N) << "\n";
        }
    }
}

//  外存条带 PageRank 迭代 
void external_stripe_pagerank(int N, int num_stripes,
                              const std::unordered_map<int,int> &outdeg) 
{
    for(int iter = 0; iter < max_iter; ++iter) {
        // 计算死节点泄漏量
        double leaked = 0.0;
        for(int sid = 0; sid < num_stripes; ++sid) {
            int start = sid * block_size;
            ifstream rf("r_stripe_" + std::to_string(sid) + ".txt");
            string line;
            int idx = 0;
            while(getline(rf, line)) {
                double val = std::stod(line);
                int node = start + idx;
                if (outdeg.find(node) == outdeg.end()) {
                    leaked += damping_factor * val;
                }
                ++idx;
            }
        }
        double leaked_share = leaked / N;
        double base = (1 - damping_factor) / N;

        double diff = 0.0;
        // 按目的条带 更新
        for(int vid = 0; vid < num_stripes; ++vid) {
            int vstart = vid * block_size;
            int vend   = min((vid+1)*block_size, N);
            int m      = vend - vstart;

            // 新条带缓冲
            vector<double> r_new(m, base + leaked_share);

            // 枚举所有源条带 uid
            for(int uid = 0; uid < num_stripes; ++uid) {
                int ustart = uid * block_size;
                int uend   = std::min((uid+1)*block_size, N);

                // 读旧 r_stripe_uid
                vector<double> r_block;
                {
                    ifstream uf("r_stripe_" + std::to_string(uid) + ".txt");
                    string l;
                    while(getline(uf,l)) {
                        r_block.push_back(std::stod(l));
                    }
                }
                // 读对应目的边 stripe_vid
                ifstream ef("stripe_" + to_string(vid) + ".txt");
                int uu, vv;
                while(ef >> uu >> vv) {
                    if (uu >= ustart && uu < uend) {
                        auto it = outdeg.find(uu);
                        if (it != outdeg.end()) {
                            int d = it->second;
                            int local_u = uu - ustart;
                            int local_v = vv - vstart;
                            r_new[local_v] += damping_factor * (r_block[local_u] / d);
                        }
                    }
                }
            }

            // 写入临时新条带并累加 diff
            string new_name = "r_new_stripe_" + std::to_string(vid) + ".txt";
            ofstream wf(new_name);
            for(double val : r_new) {
                wf << val << "\n";
            }
            // 计算 L1 差异
            {
                ifstream oldf("r_stripe_" + std::to_string(vid) + ".txt");
                ifstream newf(new_name);
                string lo, ln;
                while(getline(oldf,lo) && getline(newf,ln)) {
                    diff += fabs(stod(ln) - stod(lo));
                }
            }
        }

        // 用新文件替换旧条带
        for(int sid = 0; sid < num_stripes; ++sid) {
            fs::rename("r_new_stripe_" + std::to_string(sid) + ".txt",
                       "r_stripe_"     + std::to_string(sid) + ".txt");
        }

        cout << "Iter " << iter << " diff=" << diff << std::endl;
        if (diff < epsilon) {
            cout << "Converged.\n";
            break;
        }
    }
}

// —— k 路归并输出 Top-K —— 
void merge_top100(int N, int num_stripes, int K=100) {
    // 对每个条带内部排序并写入 sorted_sid.txt
    std::vector<std::string> sorted_files;
    for(int sid = 0; sid < num_stripes; ++sid) {
        int start = sid * block_size;
        vector<std::pair<double,int>> buf;
        ifstream rf("r_stripe_" + std::to_string(sid) + ".txt");
        string line;
        int idx = 0;
        while(getline(rf,line)) {
            buf.emplace_back(std::stod(line), start + idx);
            ++idx;
        }
        sort(buf.begin(), buf.end(),
                  [](auto &a, auto &b){ return a.first > b.first; });
        string tmp = "sorted_" + std::to_string(sid) + ".txt";
        ofstream of(tmp);
        for(auto &p : buf) {
            of << p.second << "\t" << p.first << "\n";
        }
        sorted_files.push_back(tmp);
    }

    // 4.2 k 路归并
    struct Item { double neg_score; int sid, node; };
    auto cmp = [](Item const &a, Item const &b){
        return a.neg_score > b.neg_score;
    };
    priority_queue<Item,vector<Item>,decltype(cmp)> heap(cmp);

    vector<std::ifstream> ifs;
    for(auto &fn : sorted_files) {
        ifs.emplace_back(fn);
    }
    // 初始化堆
    for(int sid = 0; sid < ifs.size(); ++sid) {
        string line;
        if (getline(ifs[sid], line)) {
            istringstream ss(line);
            int nid; double score;
            ss >> nid >> score;
            heap.push({-score, sid, nid});
        }
    }

    // 输出 Top-K
    ofstream out("res_stripe_ext.txt");
    for(int i = 0; i < K && !heap.empty(); ++i) {
        auto it = heap.top(); heap.pop();
        out << it.node << "\t" << -it.neg_score << "\n";
        // 读取该文件下一行
        string line;
        if (getline(ifs[it.sid], line)) {
            istringstream ss(line);
            int nid; double score;
            ss >> nid >> score;
            heap.push({-score, it.sid, nid});
        }
    }

    // 清理临时文件
    for(auto &fn: sorted_files) {
        fs::remove(fn);
    }
}

// —— 主流程 —— 
int main(){
    // 统计 outdeg & N
    unordered_map<int,int> outdeg;
    unordered_set<int> nodes;
    {
        ifstream fin(data_file);
        int u,v;
        while(fin >> u >> v){
            nodes.insert(u); nodes.insert(v);
            outdeg[u]++;
        }
    }
    int N = *max_element(nodes.begin(), nodes.end()) + 1;
    int num_stripes = std::ceil(double(N)/block_size);
    std::cout << "Total nodes = " << N
              << ", stripes = " << num_stripes << std::endl;

    create_edge_stripes(N, num_stripes);
    write_initial_r(N, num_stripes);
    external_stripe_pagerank(N, num_stripes, outdeg);
    merge_top100(N, num_stripes, 100);

    cout << "Done. Results in res_stripe_ext_cpp.txt\n";
    return 0;
}
