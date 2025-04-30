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
#include <chrono>
#include <thread>

using namespace std;
namespace fs = filesystem;

// —— 全局配置 —— 
const double damping_factor = 0.85;     // 跳转概率
const double epsilon    = 1e-6;     // 收敛阈值
const int    max_iter   = 100;      // 最大迭代次数
const int    block_size = 1000;     // 每个条带大小
const string data_file = "Data.txt";

// 安全删除文件
bool safe_remove_file(const string& filename, int max_retries = 3) {
    for (int i = 0; i < max_retries; ++i) {
        try {
            if (fs::exists(filename)) {
                fs::remove(filename);
                return true;
            }
            return true; // 文件不存在也算成功
        } catch (const fs::filesystem_error& e) {
            cerr << "删除文件失败: " << e.what() << "，尝试重试 (" << i + 1 << "/" << max_retries << ")" << endl;
            // 等待一段时间再重试
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }
    return false;
}

// 安全重命名文件
bool safe_rename_file(const string& old_name, const string& new_name, int max_retries = 3) {
    for (int i = 0; i < max_retries; ++i) {
        try {
            // 如果目标文件存在，先尝试删除
            if (fs::exists(new_name)) {
                fs::remove(new_name);
            }
            fs::rename(old_name, new_name);
            return true;
        } catch (const fs::filesystem_error& e) {
            cerr << "重命名文件失败: " << e.what() << "，尝试重试 (" << i + 1 << "/" << max_retries << ")" << endl;
            // 等待一段时间再重试
            this_thread::sleep_for(chrono::milliseconds(50));
        }
    }
    return false;
}

// 读取边文件，统计 outdeg 并分条带 
void create_edge_stripes(int N, int num_stripes) {
    // 创建临时目录用于保存条带文件
    string stripe_dir = "stripe_files";
    if (!fs::exists(stripe_dir)) {
        fs::create_directory(stripe_dir);
    }
    
    // 打开所有条带输出流
    vector<ofstream> stripe_fs;
    stripe_fs.reserve(num_stripes);
    for(int i = 0; i < num_stripes; ++i) {
        stripe_fs.emplace_back(stripe_dir + "/stripe_" + to_string(i) + ".txt");
    }
    // 读入边，写到对应条带
    ifstream fin(data_file);
    int u, v;
    while(fin >> u >> v) {
        int sid = v / block_size;
        if (sid >= 0 && sid < num_stripes) {
            stripe_fs[sid] << u << " " << v << "\n";
        }
    }
    for(auto &ofs : stripe_fs) ofs.close();
}

// 初始化 PageRank 向量，按条带写磁盘 —— 
void write_initial_r(int N, int num_stripes) {
    // 创建临时目录用于保存R向量文件
    string r_dir = "r_files";
    if (!fs::exists(r_dir)) {
        fs::create_directory(r_dir);
    }
    
    for(int sid = 0; sid < num_stripes; ++sid) {
        int start = sid * block_size;
        int end   = min((sid+1)*block_size, N);
        ofstream rf(r_dir + "/r_stripe_" + to_string(sid) + ".txt");
        for(int i = start; i < end; ++i) {
            rf << (1.0 / N) << "\n";
        }
    }
}

// 外存条带 PageRank 迭代 —— 
void external_stripe_pagerank(int N, int num_stripes,
                              const unordered_map<int,int> &outdeg) 
{
    string r_dir = "r_files";
    string stripe_dir = "stripe_files";
    
    for(int iter = 0; iter < max_iter; ++iter) {
        // 3.1 计算死节点泄漏量
        double leaked = 0.0;
        for(int sid = 0; sid < num_stripes; ++sid) {
            int start = sid * block_size;
            ifstream rf(r_dir + "/r_stripe_" + to_string(sid) + ".txt");
            string line;
            int idx = 0;
            while(getline(rf, line)) {
                double val = stod(line);
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
        // 3.2 按目的条带 vid 更新
        for(int vid = 0; vid < num_stripes; ++vid) {
            int vstart = vid * block_size;
            int vend   = min((vid+1)*block_size, N);
            int m      = vend - vstart;

            // 新条带缓冲
            vector<double> r_new(m, base + leaked_share);

            // 枚举所有源条带 uid
            for(int uid = 0; uid < num_stripes; ++uid) {
                int ustart = uid * block_size;
                int uend   = min((uid+1)*block_size, N);

                // 读旧 r_stripe_uid
                vector<double> r_block;
                {
                    string r_file = r_dir + "/r_stripe_" + to_string(uid) + ".txt";
                    ifstream uf(r_file);
                    if (!uf) {
                        cerr << "无法打开文件: " << r_file << endl;
                        continue;
                    }
                    string l;
                    while(getline(uf,l)) {
                        r_block.push_back(stod(l));
                    }
                }
                
                // 读对应目的边 stripe_vid
                string edge_file = stripe_dir + "/stripe_" + to_string(vid) + ".txt";
                ifstream ef(edge_file);
                if (!ef) {
                    // 如果边文件不存在，跳过
                    continue;
                }
                
                int uu, vv;
                while(ef >> uu >> vv) {
                    if (uu >= ustart && uu < uend) {
                        auto it = outdeg.find(uu);
                        if (it != outdeg.end()) {
                            int d = it->second;
                            int local_u = uu - ustart;
                            int local_v = vv - vstart;
                            if (local_u >= 0 && local_u < r_block.size() &&
                                local_v >= 0 && local_v < r_new.size()) {
                                r_new[local_v] += damping_factor * (r_block[local_u] / d);
                            }
                        }
                    }
                }
            }

            // 写入临时新条带并累加 diff
            string new_name = r_dir + "/r_new_stripe_" + to_string(vid) + ".txt";
            ofstream wf(new_name);
            for(double val : r_new) {
                wf << val << "\n";
            }
            wf.close(); // 确保文件写入完成并关闭
            
            // 计算 L1 差分
            {
                string old_name = r_dir + "/r_stripe_" + to_string(vid) + ".txt";
                ifstream oldf(old_name);
                ifstream newf(new_name);
                if (!oldf || !newf) {
                    cerr << "无法打开文件进行比较: " << old_name << " 或 " << new_name << endl;
                    continue;
                }
                string lo, ln;
                while(getline(oldf,lo) && getline(newf,ln)) {
                    diff += fabs(stod(ln) - stod(lo));
                }
            }
        }

        // 3.3 用新文件替换旧条带
        for(int sid = 0; sid < num_stripes; ++sid) {
            string old_file = r_dir + "/r_new_stripe_" + to_string(sid) + ".txt";
            string new_file = r_dir + "/r_stripe_" + to_string(sid) + ".txt";
            
            // 使用安全重命名函数
            if (!safe_rename_file(old_file, new_file)) {
                cerr << "警告: 无法重命名文件 " << old_file << " 到 " << new_file << endl;
                // 备选方案: 复制新内容到旧文件，然后删除新文件
                ifstream src(old_file, ios::binary);
                ofstream dst(new_file, ios::binary | ios::trunc);
                if (src && dst) {
                    dst << src.rdbuf();
                    src.close();
                    dst.close();
                    safe_remove_file(old_file);
                }
            }
        }

        cout << "Iter " << iter << " diff=" << diff << endl;
        if (diff < epsilon) {
            cout << "Converged.\n";
            break;
        }
    }
}

// k 路归并输出 Top-K —— 
void merge_top100(int N, int num_stripes, int K=100) {
    string r_dir = "r_files";
    string result_dir = "results";
    
    // 创建结果目录
    if (!fs::exists(result_dir)) {
        fs::create_directory(result_dir);
    }
    
    // 4.1 对每个条带内部排序并写入 sorted_sid.txt
    vector<string> sorted_files;
    for(int sid = 0; sid < num_stripes; ++sid) {
        int start = sid * block_size;
        vector<pair<double,int>> buf;
        string r_file = r_dir + "/r_stripe_" + to_string(sid) + ".txt";
        ifstream rf(r_file);
        if (!rf) {
            cerr << "无法打开文件: " << r_file << endl;
            continue;
        }
        
        string line;
        int idx = 0;
        while(getline(rf,line)) {
            buf.emplace_back(stod(line), start + idx);
            ++idx;
        }
        rf.close(); // 确保文件关闭
        
        sort(buf.begin(), buf.end(),
                  [](auto &a, auto &b){ return a.first > b.first; });
        
        string tmp = result_dir + "/sorted_" + to_string(sid) + ".txt";
        ofstream of(tmp);
        for(auto &p : buf) {
            of << p.second << "\t" << p.first << "\n";
        }
        of.close(); // 确保文件关闭
        sorted_files.push_back(tmp);
    }

    // k 路归并
    struct Item { double neg_score; int sid, node; };
    auto cmp = [](Item const &a, Item const &b){
        return a.neg_score > b.neg_score;
    };
    priority_queue<Item,vector<Item>,decltype(cmp)> heap(cmp);

    vector<ifstream> ifs;
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
    ofstream out(result_dir + "/res_stripe_ext.txt");
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
    out.close();

    // 关闭所有文件流
    for(auto &f : ifs) {
        f.close();
    }

    // 清理临时文件
    for(auto &fn: sorted_files) {
        if (!safe_remove_file(fn)) {
            cerr << "警告: 无法删除临时文件 " << fn << endl;
        }
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
    int num_stripes = ceil(double(N)/block_size);
    cout << "Total nodes = " << N
              << ", stripes = " << num_stripes << endl;

    create_edge_stripes(N, num_stripes);
    write_initial_r(N, num_stripes);
    external_stripe_pagerank(N, num_stripes, outdeg);
    merge_top100(N, num_stripes, 100);

    cout << "Done. Results in results/Res.txt\n";
    return 0;
}
 