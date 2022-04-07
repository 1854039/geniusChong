import netCDF4 as nc
import csv
import pandas as pd


def converge():
    ds = []
    path = []
    path0 = r"C:\Users\lin\Desktop\Design\Pro_Data\data5.csv"
    path.append(path0)
    path1= r"C:\Users\lin\Desktop\Design\Datasets\ssh 201601-2021\MetO-GLO-PHY-CPL-dm-SSH_1646747147247.csv"
    path.append(path1)
    # path2 = r"C:\Users\lin\Desktop\Design\Pro_Data\par\par_interp_2015.csv"
    # path3 = r"C:\Users\lin\Desktop\Design\Pro_Data\par\par_interp_2016.csv"
    # path4 = r"C:\Users\lin\Desktop\Design\Pro_Data\par\par_interp_2017.csv"
    # path5 = r"C:\Users\lin\Desktop\Design\Pro_Data\par\par_interp_2018.csv"
    # path6= r"C:\Users\lin\Desktop\Design\Pro_Data\par\par_interp_2019.csv"
    # path7 = r"C:\Users\lin\Desktop\Design\Pro_Data\par\par_interp_2020.csv"
    #
    # path8 = r"C:\Users\lin\Desktop\Design\Pro_Data\precipitation_daily\pr-wtr-2015.csv"
    # path9 = r"C:\Users\lin\Desktop\Design\Pro_Data\precipitation_daily\pr-wtr-2016.csv"
    # path10 = r"C:\Users\lin\Desktop\Design\Pro_Data\precipitation_daily\pr-wtr-2017.csv"
    # path11 = r"C:\Users\lin\Desktop\Design\Pro_Data\precipitation_daily\pr-wtr-2018.csv"
    # path12 = r"C:\Users\lin\Desktop\Design\Pro_Data\precipitation_daily\pr-wtr-2019.csv"
    # path13 = r"C:\Users\lin\Desktop\Design\Pro_Data\precipitation_daily\pr-wtr-2020.csv"
    #
    # path14 = r"C:\Users\lin\Desktop\Design\Pro_Data\wave\MetO-GLO-PHY-CPL-dm-CUR_0.csv"
    #
    # path15 = r"C:\Users\lin\Desktop\Design\Pro_Data\wind\CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE_1647578454791.csv"
    #
    # path16 = r"C:\Users\lin\Desktop\Design\Pro_Data\ssh_daily.csv"
    #
    # path17 = r"C:\Users\lin\Desktop\Design\Pro_Data\sst_daily.csv"
    # path.append(path3)
    # path.append(path4)
    # path.append(path5)
    # path.append(path6)
    # path.append(path7)
    # path.append(path8)
    # path.append(path9)
    # path.append(path10)
    # path.append(path11)
    # path.append(path12)
    # path.append(path13)
    # path.append(path14)
    # path.append(path15)
    # path.append(path16)
    # path.append(path17)



    for i in range(len(path)):
        ds.append(pd.read_csv(path[i],encoding='gb18030'))


    for i in range(len(path)):
        if i == 0:
            continue;
        data = pd.DataFrame(ds[i])
        raw = pd.DataFrame(ds[0])

        # #同时遍历当前文件和ds[0]的每一行
        # data.loc[:, "wendu_type"] = data.apply(get_wendu_type, axis=1)
        # for i,j in range(len(data.index)),range(len(raw.index)):
        # #如果坐标和时间相同，把后面那一列或者多列的值搬到这个cmem文件的后面来
        #     time=data[i][1].split(' ')
        #     raw_time=raw[j][1].split(' ')
        #     if data[i][0]==raw[j][0] and data[i][1]==raw[j][1] and time==raw_time:
        #         data.set
        df3 = pd.merge(raw, data, on=('lat', 'lon', 'time'))
        ds[0]=df3
    outputpath=r"C:\Users\lin\Desktop\Design\Pro_Data\data6.csv"
    df3.to_csv(outputpath,sep=',',index=False,header=True)

if '__name__ ==__main__':
    print("start transform!")
    # 10 11 13
    converge()
    print('Transform successfully')
