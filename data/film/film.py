import os
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# —— 0. 设置输出文件 & 加载已有进度 —— 
outfile = "imdb_2025.json"
step = 50
max_pages = 171  # 最多翻 20 页（可调整）

if os.path.exists(outfile):
    with open(outfile, "r", encoding="utf-8") as f:
        movies = json.load(f)
    pages_done = len(movies) // step
    print(f"检测到已有 {len(movies)} 条记录，已完成约 {pages_done} 页")
else:
    movies = []
    pages_done = 0

# —— 1. 配置 Selenium + Headless Chrome —— 
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("window-size=1920,1080")
options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36"
)
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# —— 2+3. 从 pages_done 开始翻页抓列表 —— 
try:
    for page in range(pages_done, max_pages):
        start = page * step + 1
        url = (
            "https://www.imdb.com/search/title/"
            "?title_type=feature&year=2025-01-01,2025-12-31"
            f"&start={start}"
        )
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "ul.ipc-metadata-list li.ipc-metadata-list-summary-item")
            )
        )
        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = soup.select("ul.ipc-metadata-list li.ipc-metadata-list-summary-item")
        if not items:
            break

        print(f"第 {page+1} 页 抓到 {len(items)} 条", end=" → ")
        for item in items:
            # 标题 & 链接（去除 ?ref_）
            title_tag = item.select_one("h3.ipc-title__text")
            link_tag  = item.select_one("a.ipc-title-link-wrapper")
            title = title_tag.get_text(strip=True) if title_tag else "未知"
            raw = link_tag["href"] if link_tag else ""
            clean = raw.split("?", 1)[0]
            link = "https://www.imdb.com" + clean if clean else None

            # 评分
            rate = item.find("span", attrs={"aria-label": re.compile(r"IMDb rating")})
            rating = rate["aria-label"].split(":",1)[1].strip() if rate and rate.has_attr("aria-label") else "null"

            # 投票数
            votes_tag = item.find("span", class_=re.compile(r"voteCount"))
            votes = int(re.sub(r"\D", "", votes_tag.get_text())) if votes_tag else 0

            # 短剧情
            pc = item.find("div", class_=re.compile(r"sttd-plot-container"))
            if pc:
                dt = pc.find("div", class_="ipc-html-content-inner-div")
                overview = dt.get_text(strip=True) if dt else "null"
            else:
                overview = "null"

            movies.append({
                "title":    title,
                "link":     link,
                "rating":   rating,
                "votes":    votes,
                "overview": overview,
            })

        # 每页抓完就写盘
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(movies, f, ensure_ascii=False, indent=2)
        print(f"累计 {len(movies)} 条已保存")

        if len(items) < step:
            break

        time.sleep(1)

    print(f"列表共抓取到 {len(movies)} 条电影记录。")

    # —— 4. 抓详情页 genres —— 
    for idx, m in enumerate(movies):
        if not m.get("link"):
            m["genres"] = []
            continue
        print(f"抓详情页 {idx+1}/{len(movies)}", end="\r")
        driver.get(m["link"])

        # # 获取 genres
        # try:
        #     WebDriverWait(driver, 10).until(
        #         EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="interests"]'))
        #     )
        #     ds = BeautifulSoup(driver.page_source, "html.parser")
        #     chips = ds.select_one('div[data-testid="interests"]')
        #     if chips:
        #         m["genres"] = [span.get_text(strip=True) for span in chips.select("span.ipc-chip__text")]
        #     else:
        #         m["genres"] = []
        # except:
        #     m["genres"] = []


        # 获取 release data
        try:

            # 1. 等待详情页标题头加载完成
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[data-testid="title-details-header"]')
                )
            )
            # 2. 解析页面
            detail_soup = BeautifulSoup(driver.page_source, "html.parser")

            # 3. 定位到 header div
            header_div = detail_soup.select_one('div[data-testid="title-details-header"]')
            release_date = None



            if header_div:
            
                # 4. 找到它后面的 metadata section
                section_div = header_div.find_next_sibling(
                    "div", {"data-testid": "title-details-section"}
                )
                if section_div:
                    
                    # 5. 在该 section 内定位 Release date 的 <li>
                    date_li = section_div.select_one('li[data-testid="title-details-releasedate"]')

                    if date_li:
                        
                        # 6. 在 content-container 内取第一个 <a>，注意中间要加空格
                        date_a = date_li.select_one(
                            "div.ipc-metadata-list-item__content-container a.ipc-metadata-list-item__list-content-item"
                        )
                        if date_a:
                            # 文本示例："January 1, 2025 (United States)"
                            raw = date_a.get_text(strip=True)
                            m["release_date"] = raw.split("(", 1)[0].strip()
                            print(f"抓详情页 {idx+1}/{len(movies)} 成功", end="\r")
                        else:
                            m["release_date"] = None
                    else: 
                        m["release_date"] = None
                else:
                    m["release_date"] = None

        except:
            m["release_date"] = None

        

    # 完成后写 final 文件
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    print("\n抓取完成，已保存至", outfile)

finally:
    driver.quit()
