import json
import argparse
import os
import os.path as osp
import tqdm

def arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--city_dir', type=str,
        help='Cityscapes LeftImg8bit directory')

    parser.add_argument('--json_dir', type=str,
        help='Top level directory for json files')

    parser.add_argument('--out_dir', type=str,
        help='Output directory to prevent overwrite',
        default='null')

    args = parser.parse_args()

    return args

def process_one_json(args, j):
    if args.out_dir == 'null':
        out_dir = args.json_dir
    else:
        out_dir = args.out_dir

    json_file = osp.join(args.json_dir ,j)
  
    try:
        json_list = json.load(open(json_file, 'rw'))
    except:
        print 'Error in loading file at: %s'%(json_file)
        return

    ip = json_list[0]['img_path']
    rel_ip = '/'.join(ip.split('/')[-3:])
    new_ip = osp.join(args.city_dir, rel_ip)
   
    if not osp.exists(new_ip):
        print 'Image at %s could not be found!'%(new_ip)
 
    for i in range(len(json_list)):
        json_list[i]['img_path'] = new_ip

    json.dump(json_list, open(osp.join(args.out_dir, j),'w'))

def check_and_create(args, d):
    ow = args.out_dir != 'null'
    if ow and not osp.exists(d):
        os.makedirs(d)


def main():
    args = arguments()
    splits = ['train', 'train_val', 'val']
    all_jsons = []

    check_and_create(args, args.out_dir)

    for s in splits:
        assert s in os.listdir(args.json_dir), 'Could not find %s split in the dataset folder!'%(s)
        d = osp.join(args.json_dir, s)

        s_dir = osp.join(args.out_dir, s)
        check_and_create(args, s_dir)
        
        cities = os.listdir(d)

        for city in cities:
            s_city_dir = osp.join(s_dir, city)
            check_and_create(args, s_city_dir)
 
            jsons = os.listdir(osp.join(d,city))
            jsons = [osp.join(s,city,j) for j in jsons]
            all_jsons.extend(jsons) 

    for j in tqdm.tqdm(all_jsons, desc='Process JSON'):
        process_one_json(args, j)

if __name__ == '__main__':
    main()
