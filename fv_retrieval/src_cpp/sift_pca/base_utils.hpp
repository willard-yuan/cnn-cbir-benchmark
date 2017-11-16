
#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <glob.h>
#include <math.h>
#include <string>
#include <sstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

/**********************************************
 * File processing functions
 **********************************************/

// Split string by special char, such as by space
// Usage: split("hello world", ' ')
std::vector<std::string> split_string(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

// Split string by special char, such as by space
std::vector<float> split_float(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<float> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(stof(item));
    }
    return tokens;
}


// Get the files full paths from directory
std::vector<std::string> glob_vector(const std::string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    std::vector<std::string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        auto tmp = std::string(glob_result.gl_pathv[i]);
        files.push_back(tmp);
    }
    std::sort(files.begin(), files.end());
    globfree(&glob_result);
    return files;
}

// Get base name from a full path
// Usage: base_name("/xxx/xxx/123.jpg")
// Output: 123.jpg
std::string base_name(std::string const & path, std::string const & delims){
    return path.substr(path.find_last_of(delims) + 1);
}

// Remove extension from a base name
// Usage: remove_extension("123.jpg")
// Output: 123
std::string remove_extension(std::string const & filename){
    typename std::string::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != std::string::npos ? filename.substr(0, p) : filename;
}

// Split string by substring
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c){
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
        
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}


/**********************************************
 * Math processing functions
 **********************************************/

// Compute cosine similarity between two vectors
float cosine_similarity(std::vector<float> &v1, std::vector<float> &v2){
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    dot = std::inner_product( v1.begin(), v1.end(), v2.begin(), 0.0 );
    denom_a = std::inner_product( v1.begin(), v1.end(), v1.begin(), 0.0 );
    denom_b = std::inner_product( v2.begin(), v2.end(), v2.begin(), 0.0 );
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

// Normalize vector by L2 norm
std::vector<float> nomalize_vecotor(std::vector<float> &v){
    std::vector<float> v_norm;
    float denom_v = std::inner_product( v.begin(), v.end(), v.begin(), 0.0 );
    if (0.0 != denom_v){
        for (auto it = v.begin(); it != v.end(); it++){
            float tmp = (*it)/sqrt(denom_v);
            v_norm.push_back(tmp);
        }
    }else{
        v_norm = v;
    }
    return v_norm;
}

#endif /* utils_hpp */
