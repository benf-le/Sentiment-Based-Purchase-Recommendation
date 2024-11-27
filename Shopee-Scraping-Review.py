import requests
import csv
from datetime import datetime


def shopee(url):
    shop_name = url.split('/')[-1]
    url = f'https://shopee.co.id/api/v4/shop/get_shop_detail?sort_sold_out=0&username={shop_name}'
    req = requests.get(url)
    data_shop = req.json()
    shop_id = data_shop['data']['shopid']
    user_id = data_shop['data']['userid']
    count = 0
    result = []
    while True:
        try:
            count += 6
            url = f'https://shopee.co.id/api/v4/seller_operation/get_shop_ratings?limit=6&offset={count}&shop_id={shop_id}&user_id={user_id}'
            req = requests.get(url)
            data_req = req.json()
            for value in data_req['data']:
                data_result = {

                    'review': value['comment'],
                    'rating': value['rating_star'],
                }
                result.append(data_result)
        except KeyError:
            break

    # save to csv
    keys = result[0].keys()
    with open(f'shoope_rating_{shop_name}.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result)


if __name__ == '__main__':
    # silakan ganti url_shope jika ingin mengambil data review dan rating dari toko lain
    url_shop = 'https://shopee.vn/somebymi.vn'
    shopee(url_shop)
