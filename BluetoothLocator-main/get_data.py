import json
import numpy as np
import pandas as pd
import os

def load_beacons(beacon_path):
    with open(beacon_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def rssi_from_distance(distance, tx_power=-53.97, path_loss_exponent=2.36):
    if distance <= 0:
        return tx_power
    rssi = tx_power - 10 * path_loss_exponent * np.log10(distance)
    return int(round(rssi))

def interpolate_points(p1, p2, num):
    return [(p1[0] + (p2[0] - p1[0]) * i / (num - 1),
             p1[1] + (p2[1] - p1[1]) * i / (num - 1)) for i in range(num)]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    beacon_path = os.path.join(base_dir, 'beacon', 'beacon_database.json')
    output_path = os.path.join(base_dir, 'data', 'fakedata.xlsx')
    beacons = load_beacons(beacon_path)
    # 选定信标
    p1 = np.array([
        (beacons['0000000002AD']['longitude'] + beacons['00000000093C']['longitude']) / 2,
        (beacons['0000000002AD']['latitude'] + beacons['00000000093C']['latitude']) / 2
    ])
    p2 = np.array([
        (beacons['000000000498']['longitude'] + beacons['000000000AB9']['longitude']) / 2,
        (beacons['000000000498']['latitude'] + beacons['000000000AB9']['latitude']) / 2
    ])
    # 生成轨迹
    steps = 10
    points = interpolate_points(p1, p2, steps)
    # 生成数据
    records = []
    for idx, (lon, lat) in enumerate(points):
        for mac, info in beacons.items():
            # 计算距离（米）
            d = haversine(lat, lon, info['latitude'], info['longitude'])
            rssi = rssi_from_distance(d)
            records.append({
                'id': idx,
                'device_id': 'FAKE01',
                'mac': mac,
                'rssi': rssi,
                'rotation': 0,
                'timestamp': f'step_{idx}'
            })
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name='bluetooth_position_data')
    print(f'虚拟蓝牙数据已生成: {output_path}，sheet名为bluetooth_position_data')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

if __name__ == '__main__':
    main()
