# 爬取通过输入数字选择爬取百度或搜狗图片
import re
import requests
import os
import hashlib

md5_list = []
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82'


def get_md5(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    files = os.listdir(path)
    print(f'本地存在{len(files)}张关于{path.split(r"/")[-1]}的图片')
    for i in files:
        with open(path + '/' + i, 'rb') as f:
            md5_string = hashlib.md5(f.read()).hexdigest()
            md5_list.append(md5_string)


def get_img(name, img_mode, x, path):
    img_list = []
    print('正在获取图片列表...')
    for i in range(int(x)):
        if img_mode == '1':
            url = 'https://image.baidu.com/search/acjson'
            params = {'tn': 'resultjson_com', 'ipn': 'rj', 'word': name, 'queryWord': name, 'ie': 'utf-8',
                      'oe': 'utf-8', 'pn': 50 * i, 'rn': 50, }
            headers = {'Accept-Language': 'zh-CN,zh;q=0.9', 'Host': 'image.baidu.com', 'User-Agent': user_agent}
            try:
                response = requests.get(url, params=params, headers=headers).text
                img_url_list = re.findall(r'"hoverURL":"(http.+?)"', response)
                img_list += img_url_list
            except:
                continue
        else:
            url = f'https://pic.sogou.com/napi/pc/searchList?mode=1&start={i * 50}&xml_len=50&query={name}'
            headers = {'User-Agent': user_agent}
            try:
                response = requests.get(url, headers=headers).json()
                for j in response.get('data').get('items'):
                    if j.get('picUrl'):
                        img_list.append(j['picUrl'])
            except:
                continue
    print(f'共获取到{len(img_list)}张关于{name}的图片')
    download_img(img_list, name, path)


def download_img(img_list, name, path):
    repeat_num = 0
    success_num = 0
    false_num = 0
    for i, item in enumerate(img_list):
        print(f'正在下载第{i + 1}张图片...')
        try:
            img = requests.get(item, timeout=5)
            suffix = img.headers['Content-Type'].split('/')[1]
            if suffix not in ['jpg', 'jpeg']:
                print(f'格式为{suffix}，跳过下载！')
                repeat_num += 1
                continue
            md5_string = hashlib.md5(img.content).hexdigest()
            if md5_string not in md5_list:
                with open(f'{path}/{md5_string}.{suffix}', 'wb') as file:
                    file.write(img.content)
                success_num += 1
                md5_list.append(md5_string)
            else:
                print('该图片已存在，跳过下载！')
                repeat_num += 1
        except:
            print(f'第{i + 1}张图片下载失败！')
            false_num += 1
    print(f'本次共下载{len(img_list)}张图片，成功下载{success_num}张，跳过重复的{repeat_num}张，下载失败{false_num}张，'
          f'当前文件夹共有{len(md5_list)}张{name}图片')


def main():
    name = input('请输入你要下载的图片类型：')
    img_mode = input('请选择下载图片的网站，1代表 “百度” ，2代表 “搜狗”，如需退出请输入0：')
    x = input('请输入你要下载的数量，1代表50，2代表100，以此类推：')
    path = '../resources/flower_photos/' + name
    if img_mode == '0':
        return
    get_md5(path)
    get_img(name, img_mode, x, path)


if __name__ == '__main__':
    main()
