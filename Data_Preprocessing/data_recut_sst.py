import xarray as xr
import netCDF4 as nc
path1=r"C:\Users\lin\Desktop\Design\Datasets\ssh 201601-2021\MetO-GLO-PHY-CPL-dm-SSH_1646747147247.nc"
olr=xr.open_dataset(path1)#读取数据

pa1=r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pr-wtr-2015.nc"
pa2=r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pr-wtr-2016.nc"

pa3=r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pr-wtr-2017.nc"

pa4=r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pr-wtr-2018.nc"

pa5=r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pr-wtr-2019.nc"

pa6=r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pr-wtr-2020.nc"
da=[]

da.append(xr.open_dataset(pa1,engine = 'netcdf4'))
da.append(xr.open_dataset(pa2,engine = 'netcdf4',decode_cf=False))
da.append(xr.open_dataset(pa3,engine = 'netcdf4',decode_cf=False))
da.append(xr.open_dataset(pa4,engine = 'netcdf4',decode_cf=False))
da.append(xr.open_dataset(pa5,engine = 'netcdf4',decode_cf=False))
da.append(xr.open_dataset(pa6,engine = 'netcdf4',decode_cf=False))


# par1=da[0]['pr_wtr']
# par2=da[1]['pr_wtr']
# par3=da[2]['pr_wtr']
# par4=da[3]['pr_wtr']
# par5=da[4]['pr_wtr']
# par6=da[5]['pr_wtr']
datasets=[]

datasets.append(nc.Dataset(pa1))
datasets.append(nc.Dataset(pa2))
datasets.append(nc.Dataset(pa3))
datasets.append(nc.Dataset(pa4))
datasets.append(nc.Dataset(pa5))
datasets.append(nc.Dataset(pa6))

# 获取相应数组集合--纬度经度温度深度
# for d in range(len(datasets)):
#     lat_set = datasets[d].variables['lat'][:]
#     lon_set =datasets[d].variables['lon'][:]
#     time =datasets[d].variables['time'][:]
#     #real_time = nc.num2date(time, 'days since 1800-1-1 00:00:0.0')
#     zos =datasets[d].variables['pr_wtr'][:]
# for i in range(len(da)):

v = da[0]['pr_wtr'][:]  # prec为变量内容（降雨）
nc=v.loc[:]
nc_coordinate = nc.loc[:,25:30, 120:122.5]  # 截取经纬度

ds1 = xr.Dataset({'pr_wtr': nc_coordinate})
ds1.to_netcdf(r"C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily"+"\\"+"pre_daily_2015.nc")

#以下这一步就是插值的过程
#sst= sst.interp(lat=olr.lat.values, lon=olr.lon.values)
# p1= par1.interp(lat=olr.lat.values, lon=olr.lon.values)
# p2= par2.interp(lat=olr.lat.values, lon=olr.lon.values)
# p3= par3.interp(lat=olr.lat.values, lon=olr.lon.values)
# p4= par4.interp(lat=olr.lat.values, lon=olr.lon.values)
# p5= par5.interp(lat=olr.lat.values, lon=olr.lon.values)
# p6= par6.interp(lat=olr.lat.values, lon=olr.lon.values)
# p1.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pre_interp_2015.nc')
# p2.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pre_interp_2016.nc')
# p3.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pre_interp_2017.nc')
# p4.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pre_interp_2018.nc')
# p5.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pre_interp_2019.nc')
# p6.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\pre 2015-2020 daily\pre_interp_2020.nc')
#sst.to_netcdf(r'C:\Users\lin\Desktop\Design\Datasets\sst 2015-2020 daily\sst_interp.nc')