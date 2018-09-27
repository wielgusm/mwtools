import pandas as pd
from eat.io import uvfits
from eat.inspect import utils as ut
import os,sys,importlib

def import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,pipeline_name='hops',tavg='scan',exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=True,filend=".uvfits",incoh_avg=False,out_type='hdf',rescale_noise=False,polrep=None):

    if not os.path.exists(path_out):
        os.makedirs(path_out) 
    df = pd.DataFrame({})
    for band in bandL:  
        for expt in exptL:
            path0 = path_data_0+pipeline_name+'-'+band+'/'+data_subfolder+str(expt)+'/'
            for filen in os.listdir(path0):
                if filen.endswith(filend): 
                    print('processing ', filen)
                    try:
                        df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='',band=band,round_s=0.1,only_parallel=only_parallel,rescale_noise=rescale_noise,polrep=polrep)
                        if 'std_by_mean' in df_foo.columns:
                            df_foo.drop('std_by_mean',axis=1,inplace=True)
                        df_foo['std_by_mean'] = df_foo['amp']
                        if incoh_avg==False:
                            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                        else:
                            df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                        df = pd.concat([df,df_scan.copy()],ignore_index=True)
                        df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
                    except: pass
                else: pass 
    df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    df['source'] = list(map(str,df['source']))
    if len(bandL)==1:
        out_name=out_name+'_'+bandL[0]        
    if out_type=='hdf':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
    elif out_type=='pic':
        df.to_pickle(path_out+out_name+'.pic')
    elif out_type=='both':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
        df.to_pickle(path_out+out_name+'.pic')
    else: return df
    

def import_uvfits_folder(path_folder,path_vex,path_out,out_name,pipeline_name='hops',tavg='scan',
    force_singlepol='no',band='none',only_parallel=True,filend=".uvfits",incoh_avg=False,out_type='hdf',
    rescale_noise=False,polrep=None,polrep_path_ehtwork=''):

    #if polrep_path_ehtwork!='':
    #    sys.path.remove('/usr/local/src/ehtim')
    #    #sys.path.append('/home/maciek/polrep/eht-polrep')
    #    sys.path.append(polrep_path_ehtwork)
    #    importlib.reload(eh)
    #    sys.path.append('/usr/local/src/ehtim')

    if not os.path.exists(path_out):
        os.makedirs(path_out) 
    df = pd.DataFrame({})
    path0 = path_folder
    for filen in os.listdir(path0):
        if filen.endswith(filend): 
            print('processing ', filen)
            #try:
            df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol=force_singlepol,band=band,round_s=0.1,
            only_parallel=only_parallel,rescale_noise=rescale_noise,polrep=polrep)
            if 'std_by_mean' in df_foo.columns:
                df_foo.drop('std_by_mean',axis=1,inplace=True)
            df_foo['std_by_mean'] = df_foo['amp']
            if incoh_avg==False:
                df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
            else:
                df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
            df = pd.concat([df,df_scan.copy()],ignore_index=True)
            df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
            #except ValueError: pass
        else: pass
    print(df.columns) 
    df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    df['source'] = list(map(str,df['source']))
    if band!='none':
        out_name=out_name+'_'+band        
    if out_type=='hdf':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
    elif out_type=='pic':
        df.to_pickle(path_out+out_name+'.pic')
    elif out_type=='both':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
        df.to_pickle(path_out+out_name+'.pic')
    else: return df


def import_uvfits_set_netcal(path_data_0,data_subfolder,path_vex,path_out,out_name,tavg='scan',exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],filend="netcal.uvfits",incoh_avg=False,out_type='hdf',polrep=None):

    if not os.path.exists(path_out):
        os.makedirs(path_out) 
    df = pd.DataFrame({})
    #first all LL
    for band in bandL:  
        for expt in exptL:
            path0 = path_data_0+'hops-'+band+'/'+data_subfolder+str(expt)+'/'
            for filen in os.listdir(path0):
                if filen.endswith(filend): 
                    df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='no',band=band,round_s=0.1,only_parallel=True,polrep=polrep)
                    if tavg!=-1:
                        if incoh_avg==False:
                            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                        else:
                            df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                        df = pd.concat([df,df_scan],ignore_index=True) 
                    else:
                        print('no averaging')
                        df = pd.concat([df,df_foo],ignore_index=True)  
    df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    df['source'] = list(map(str,df.source))
    if out_type=='hdf':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
    elif out_type=='pic':
        df.to_pickle(path_out+out_name+'.pic')
    else: return df


def import_alist(path_data_0,data_subfolder,filen,path_out,out_name,bandL=['lo','hi']):
    from eat.io import hops
    if not os.path.exists(path_out):
        os.makedirs(path_out) 
    df = pd.DataFrame({})

    for band in bandL:  
            path_data = path_data_0+'hops-'+band+'/'+data_subfolder+'data/'+filen
            df_foo = hops.read_alist(path_data)
            df_foo['band'] = band
            df = pd.concat([df,df_foo],ignore_index=True)
            
    df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    df.drop(list(df[df.baseline.str[0]==df.baseline.str[1]].index.values),inplace=True) 
    df['phase'] = df['resid_phas']
    df['amp'] = (1.e-4)*df['amp']
    df['sigma'] = df['amp']/df['snr']
    df['scan_id']=list(map(lambda x: dict_scan_id[x],df['scan_id']))

    #df.to_pickle(path_out+out_name+'.pic')
    df['source'] = list(map(str,df.source))
    if out_type=='hdf':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
    elif out_type=='pic':
        df.to_pickle(path_out+out_name+'.pic')
    else: return df



dict_scan_id={'094-2231': 0,
 '094-2242': 1,'094-2253': 2,'094-2304': 3,'094-2319': 4,'094-2330': 5,'094-2344': 6,
 '094-2355': 7,'095-0009': 8,'095-0020': 9,'095-0031': 10,'095-0046': 11,'095-0100': 12,
 '095-0110': 13,'095-0125': 14,'095-0138': 15,'095-0149': 16,'095-0159': 17,'095-0214': 18,
 '095-0225': 19,'095-0236': 20,'095-0251': 21,'095-0258': 22,'095-0310': 23,'095-0325': 24,
 '095-0332': 25,'095-0345': 26,'095-0359': 27,'095-0408': 28,'095-0415': 29,'095-0428': 30,
 '095-0435': 31,'095-0450': 32,'095-0500': 33,'095-0511': 34,'095-0519': 35,'095-0530': 36,
 '095-0538': 37,'095-0549': 38,'095-0557': 39,'095-0612': 40,'095-0620': 41,'095-0631': 42,
 '095-0639': 43,'095-0650': 44,'095-0658': 45,'095-0709': 46,'095-0728': 47,'095-0736': 48,
 '095-0747': 49,'095-0755': 50,'095-0806': 51,'095-0814': 52,'095-0829': 53,'095-0839': 54,
 '095-0849': 55,'095-0859': 56,'095-0908': 57,'095-0923': 58,'095-0935': 59,'095-0942': 60,
 '095-0954': 61,'095-1005': 62,'095-1012': 63,'095-1025': 64,'095-1036': 65,'095-1045': 66,
 '095-1052': 67,'095-1105': 68,'095-1118': 69,'095-1129': 70,'095-1136': 71,'095-1148': 72,
 '095-1156': 73,'095-1209': 74,'095-1223': 75,'095-1230': 76,'095-1243': 77,'095-1257': 78,
 '095-1310': 79,'095-1324': 80,'095-1331': 81,'095-1344': 82,'095-1355': 83,'095-1402': 84,
 '095-1419': 85,'095-1431': 86,'095-1438': 87,'095-1449': 88,'095-1458': 89,'095-1505': 90,
 '095-1513': 91,'095-1526': 92,'095-1533': 93,'095-1547': 94,'095-1558': 95,'095-1605': 96,
 '095-1612': 97,'095-1619': 98,'095-1628': 99,'095-1635': 100,'095-1642': 101,'095-1649': 102,
 '095-1656': 103,'095-1703': 104,'096-0046': 105,'096-0052': 106,'096-0104': 107,'096-0110': 108,
 '096-0122': 109,'096-0128': 110,'096-0140': 111,'096-0146': 112,'096-0158': 113,'096-0204': 114,
 '096-0218': 115,'096-0224': 116,'096-0236': 117,'096-0242': 118,'096-0254': 119,'096-0300': 120,
 '096-0312': 121,'096-0318': 122,'096-0332': 123,'096-0338': 124,'096-0350': 125,'096-0356': 126,
 '096-0408': 127,'096-0414': 128,'096-0426': 129,'096-0432': 130,'096-0446': 131,'096-0452': 132,
 '096-0502': 133,'096-0508': 134,'096-0518': 135,'096-0524': 136,'096-0534': 137,'096-0540': 138,
 '096-0554': 139,'096-0600': 140,'096-0610': 141,'096-0616': 142,'096-0626': 143,'096-0632': 144,
 '096-0642': 145,'096-0648': 146,'096-0702': 147,'096-0708': 148,'096-0718': 149,'096-0724': 150,
 '096-0734': 151,'096-0740': 152,'096-0750': 153,'096-0756': 154,'096-0817': 155,'096-0824': 156,
 '096-0835': 157,'096-0847': 158,'096-0854': 159,'096-0905': 160,'096-0918': 161,'096-0926': 162,
 '096-0938': 163,'096-0949': 164,'096-1000': 165,'096-1012': 166,'096-1019': 167,'096-1030': 168,
 '096-1041': 169,'096-1048': 170,'096-1059': 171,'096-1110': 172,'096-1121': 173,'096-1132': 174,
 '096-1144': 175,'096-1151': 176,'096-1202': 177,'096-1214': 178,'096-1221': 179,'096-1237': 180,
 '096-1248': 181,'096-1300': 182,'096-1307': 183,'096-1318': 184,'096-1330': 185,'096-1337': 186,
 '096-1353': 187,'096-1404': 188,'096-1415': 189,'096-1425': 190,'096-1437': 191,'096-1444': 192,
 '096-1453': 193,'096-1505': 194,'096-1512': 195,'096-1522': 196,'096-1533': 197,'096-1541': 198,
 '096-1551': 199,'096-1601': 200,'096-1611': 201,'097-0401': 202,'097-0414': 203,'097-0423': 204,
 '097-0433': 205,'097-0446': 206,'097-0455': 207,'097-0506': 208,'097-0518': 209,'097-0529': 210,
 '097-0541': 211,'097-0553': 212,'097-0602': 213,'097-0612': 214,'097-0625': 215,'097-0634': 216,
 '097-0646': 217,'097-0700': 218,'097-0709': 219,'097-0720': 220,'097-0729': 221,'097-0747': 222,
 '097-0804': 223,'097-0812': 224,'097-0828': 225,'097-0836': 226,'097-0848': 227,'097-0858': 228,
 '097-0904': 229,'097-0913': 230,'097-0926': 231,'097-0937': 232,'097-0950': 233,'097-0958': 234,
 '097-1011': 235,'097-1019': 236,'097-1029': 237,'097-1035': 238,'097-1046': 239,'097-1057': 240,
 '097-1108': 241,'097-1117': 242,'097-1130': 243,'097-1136': 244,'097-1146': 245,'097-1158': 246,
 '097-1204': 247,'097-1220': 248,'097-1231': 249,'097-1237': 250,'097-1250': 251,'097-1303': 252,
 '097-1309': 253,'097-1326': 254,'097-1337': 255,'097-1343': 256,'097-1356': 257,'097-1407': 258,
 '097-1420': 259,'097-1428': 260,'097-1439': 261,'097-1450': 262,'097-1501': 263,'097-1507': 264,
 '097-1518': 265,'097-1531': 266,'097-1539': 267,'097-1547': 268,'097-1555': 269,'097-1608': 270,
 '097-1618': 271,'097-1626': 272,'097-1636': 273,'097-1646': 274,'097-1654': 275,'097-1704': 276,
 '097-1717': 277,'097-1725': 278,'097-1735': 279,'097-1745': 280,'097-1753': 281,'097-1803': 282,
 '097-1813': 283,'097-1821': 284,'097-1834': 285,'097-1844': 286,'097-1854': 287,'097-1904': 288,
 '097-1914': 289,'097-1924': 290,'097-1937': 291,'097-1946': 292,'097-1953': 293,'097-2000': 294,
 '097-2009': 295,'097-2018': 296,'097-2027': 297,'097-2039': 298,'099-2317': 299,'099-2328': 300,
 '099-2337': 301,'099-2348': 302,'099-2359': 303,'100-0012': 304,'100-0023': 305,'100-0035': 306,
 '100-0046': 307,'100-0057': 308,'100-0108': 309,'100-0123': 310,'100-0134': 311,'100-0145': 312,
 '100-0159': 313,'100-0209': 314,'100-0221': 315,'100-0232': 316,'100-0243': 317,'100-0252': 318,
 '100-0301': 319,'100-0309': 320,'100-0321': 321,'100-0330': 322,'100-0340': 323,'100-0359': 324,
 '100-0407': 325,'100-0416': 326,'100-0426': 327,'100-0437': 328,'100-0445': 329,'100-0453': 330,
 '100-0504': 331,
 '100-0517': 332,
 '100-0525': 333,
 '100-0533': 334,
 '100-0544': 335,
 '100-0555': 336,
 '100-0603': 337,
 '100-0611': 338,
 '100-0631': 339,
 '100-0639': 340,
 '100-0649': 341,
 '100-0657': 342,
 '100-0707': 343,
 '100-0715': 344,
 '100-0725': 345,
 '100-0738': 346,
 '100-0746': 347,
 '100-0756': 348,
 '100-0804': 349,
 '100-0814': 350,
 '100-0822': 351,
 '100-0830': 352,
 '100-0843': 353,
 '100-0853': 354,
 '100-0901': 355,
 '100-0909': 356,
 '100-0917': 357,
 '100-0925': 358,
 '100-0933': 359,
 '100-0941': 360,
 '100-0949': 361,
 '100-0957': 362,
 '100-1012': 363,
 '100-1021': 364,
 '100-1032': 365,
 '100-1043': 366,
 '100-1052': 367,
 '100-1101': 368,
 '100-1114': 369,
 '100-1123': 370,
 '100-1141': 371,
 '100-1150': 372,
 '100-1203': 373,
 '100-1216': 374,
 '100-1229': 375,
 '100-1241': 376,
 '100-1254': 377,
 '100-1307': 378,
 '100-1316': 379,
 '100-1329': 380,
 '100-1342': 381,
 '100-1354': 382,
 '100-1407': 383,
 '100-1420': 384,
 '100-1429': 385,
 '100-1442': 386,
 '100-1450': 387,
 '100-1459': 388,
 '100-1506': 389,
 '100-2216': 390,
 '100-2225': 391,
 '100-2234': 392,
 '100-2243': 393,
 '100-2252': 394,
 '100-2304': 395,
 '100-2316': 396,
 '100-2325': 397,
 '100-2334': 398,
 '100-2343': 399,
 '100-2352': 400,
 '101-0001': 401,
 '101-0013': 402,
 '101-0032': 403,
 '101-0041': 404,
 '101-0050': 405,
 '101-0102': 406,
 '101-0115': 407,
 '101-0124': 408,
 '101-0138': 409,
 '101-0150': 410,
 '101-0159': 411,
 '101-0212': 412,
 '101-0224': 413,
 '101-0233': 414,
 '101-0248': 415,
 '101-0300': 416,
 '101-0309': 417,
 '101-0322': 418,
 '101-0334': 419,
 '101-0343': 420,
 '101-0358': 421,
 '101-0410': 422,
 '101-0419': 423,
 '101-0432': 424,
 '101-0444': 425,
 '101-0453': 426,
 '101-0515': 427,
 '101-0524': 428,
 '101-0535': 429,
 '101-0546': 430,
 '101-0555': 431,
 '101-0606': 432,
 '101-0620': 433,
 '101-0629': 434,
 '101-0640': 435,
 '101-0651': 436,
 '101-0700': 437,
 '101-0711': 438,
 '101-0725': 439,
 '101-0736': 440,
 '101-0745': 441,
 '101-0754': 442,
 '101-0803': 443,
 '101-0812': 444,
 '101-0821': 445,
 '101-0830': 446,
 '101-0839': 447,
 '101-0900': 448,
 '101-0913': 449,
 '101-0920': 450,
 '101-0933': 451,
 '101-0947': 452,
 '101-0954': 453,
 '101-1007': 454,
 '101-1018': 455,
 '101-1036': 456,
 '101-1043': 457,
 '101-1053': 458,
 '101-1106': 459,
 '101-1120': 460,
 '101-1127': 461,
 '101-1139': 462,
 '101-1152': 463,
 '101-1208': 464,
 '101-1215': 465,
 '101-1227': 466,
 '101-1241': 467,
 '101-1248': 468,
 '101-1259': 469,
 '101-1313': 470,
 '101-1324': 471,
 '101-1335': 472,
 '101-1348': 473,
 '101-1355': 474,
 '101-1406': 475,
 '101-1419': 476,
 '101-1431': 477,
 '101-1442': 478,
 '101-1453': 479,
 '101-1505': 480,
 '101-1512': 481,
 '101-1518': 482}