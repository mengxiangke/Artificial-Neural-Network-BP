/*
Author:山野莽夫
Web：https://www.shanyemangfu.com
version:3.0
*/
#include <iostream>
#include<math.h>
#include<fstream>
#include<algorithm>
#include<ctime>
#include <iomanip>
using namespace std;
//定义函数
double f(double x, int kind);//激活函数
double df(double x, int kind);//激活函数的导数
void readdata();//读取数据
void initial();//初始化
void save();//保存模型
void load();//加载模型
void normalall();//全部数据归一化
bool dobatchtrain(int traintime, double acc, int& times);//
double batchtrain(); //小批量随机梯度下降，单次更新
void batchpos(int begin, int batchsize);//小批量正向计算
void batchnega(int begin, int batchsize);//小批量反向更新
void btest();//测试
//参数
constexpr int hidenum = 2;//隐藏层层数;
constexpr int num[hidenum + 1 + 1 + 1] = { 5,4,5,5,3 };//各层神经元个数0位置存储最大值
constexpr double e = 2.718281828459;
constexpr int tnum = 150;//训练样本数目
constexpr double tau = 0.001;//学习率
constexpr int trainnum = 40;//训练样本数目
constexpr int batchsize = 5;//批大小，要设置为可以被样本数目整除，虽说可以代码处理，但是懒得处理。
//变量
double x[num[1] + 1][tnum + 1];//输入
double y[num[hidenum + 2] + 1][tnum + 1];//期望输出
double nmx[num[1] + 1][tnum + 1];//归一化后的输入
double nmy[num[hidenum + 2] + 1][tnum + 1];//归一化后的期望输出
double w[hidenum + 2][num[0] + 1][num[0] + 1];//层数/k层顺序/k-1层顺序  |权重
double bb[hidenum + 2 + 1][num[0] + 1];//偏置
double bd[hidenum + 2 + 1][num[0] + 1][batchsize + 1];//误差
double by[num[hidenum + 2] + 1][batchsize + 1];//预测输出值
double bz[hidenum + 2 + 1][num[0] + 1][batchsize + 1];//净输出
double ba[hidenum + 2 + 1][num[0] + 1][batchsize + 1];//输出
int r = 1;//控制样本
ofstream rout("rout.txt");//迭代误差


int main()
{
	//std::cout << "Hello World!\n";
	//ifstream fin("data.txt");
	/*ofstream fout("fout.txt");
	ofstream pout("pout.txt");*/
	int times;//实际训练次数
	int traintime = 50000000;//最大训练次数
	double acc = 0.1;//精度
	readdata();//读取训练集
	initial();//初始化
	normalall();
	int option;
	cout << "请选择操作（输入1|2）" << "1:加载模型并测试\t" << "2：训练新模型" << endl;
	cin >> option;
	if (option==1) {
		load();
		btest();
	}
	else if (option == 2) {
		cout << "请输入Y或者N来选择是否保存模型" << endl;
		char issave;
		cin >> issave;
		if (issave == 'Y')
			cout << "你选择保存模型" << endl;
		if (dobatchtrain(traintime, acc, times))
			cout << "迭代" << times << "次训练成功" << endl;
		else
			cout << times << "训练失败，结果不收敛，请调整参数或者训练集" << endl;
		btest();
		if (issave == 'Y')
			save();
		
	}
	else cout << "输入不正确" << endl;
}
void save() {
	ofstream model("model.txt");
	for (int l = 2; l <= hidenum + 2; l++) {
		//model << l <<"\t"<<b[l]<< endl;//层号、偏置
		model << l << endl;//层号、偏置
		for (int i = 1; i <= num[l]; i++) {
			for (int j = 1; j <= num[l - 1]; j++) {
				model << w[l - 1][i][j] << "\t";//权重
			}
			model << bb[l][i] << endl;
		}
	}
	/*for (int l = 1; l <= hidenum + 1; l++) {
		wout << l << endl;
		for (int i = 1; i <= num[l+1]; i++) {

			for (int j = 1; j <= num[l]; j++) {
				wout << w[l][i][j] << "\t";
			}
			wout << endl;
		}
	}*/

}
void load() {
	ifstream readmodel("model.txt");
	for (int l = 2; l <= hidenum + 2; l++) {
		//readmodel >> l >> b[l] ;//层号、偏置
		readmodel >> l;//层号
		for (int i = 1; i <= num[l]; i++) {
			for (int j = 1; j <= num[l - 1]; j++) {
				readmodel >> w[l - 1][i][j];//权重
			}
			readmodel >> bb[l][i];
		}
	}
}
void readdata() {
	ifstream fin("data.txt");
	ofstream fout("fout.txt");
	int temp;
	/*fin >> temp;
	cout << temp;*/
	for (int i = 1; i <= tnum; i++) {
		for (int j = 1; j <= num[1]; j++) {
			fin >> x[j][i];
			//cout << x[j][i] << "\t";
		}
		//fin >> sy[i];
		for (int j = 1; j <= num[hidenum + 2]; j++) {
			fin >> y[j][i];
		}
	}
	for (int i = 0; i <= num[hidenum + 2]; i++) {
		for (int j = 0; j <= tnum; j++)
			fout << y[i][j] << "\t";
		fout << endl;
	}
	//测试真实值
}
double f(double x, int kind) {
	if (kind == 1) {
		return 1 / (1 + pow(e, -x));
	}
	else if (kind == 2) {
		//return max(0.0, x);
		if (x > 0)
			return x;
		else
			return 0;
	}
	else if (kind == 3) {
		return (pow(e, x) - pow(e, -x)) / (pow(e, x) + pow(e, -x));
	}
	else if (kind == 4) {
		return x;
	}
	else return 1 / (1 + pow(e, -x));
}
double df(double x, int kind) {
	if (kind == 1)
		return f(x, 1) * (1 - f(x, 1));
	else if (kind == 2) {
		if (x > 0)
			return 1;
		else
			return 0;
	}
	else if (kind == 3) {
		return 1 - f(x, 3) * f(x, 3);
	}
	else if (kind == 4) {
		return 1;
	}
	else return f(x, 1) * (1 - f(x, 1));
}
void initial() {//初始化权值和偏置	
	for (int i = 1; i <= hidenum + 2; i++) {
		for (int j = 1; j <= num[0]; j++) {
			bb[i][j] = 0;
		}
	}
	srand(time(NULL));
	for (int l = 2; l <= hidenum + 2; l++) {//生成-1--1随机数
		//cout << l - 1 << endl;
		for (int i = 1; i <= num[l]; i++) {
			for (int j = 1; j <= num[l - 1]; j++) {
				w[l - 1][i][j] = 2.0 * rand() / RAND_MAX - 1;
				//cout << w[l - 1][i][j];
			}
			//cout << endl;
		}
	}
}
void btest() {
	ofstream btout("btout.txt");
	int count = 0;
	for (int i = trainnum + 1; i <= tnum; i += batchsize) {
		batchpos(i, batchsize);
		for (int n = 1; n <= batchsize; n++) {
			for (int j = 1; j <= num[1]; j++) {
				cout << x[j][i + n - 1] << "\t";
				btout << x[j][i + n - 1] << "\t";
			}
			for (int j = 1; j <= num[hidenum + 2]; j++) {
				cout << y[j][i + n - 1] << "\t";
				btout << y[j][i + n - 1] << "\t";
			}			
			for (int j = 1; j <= num[hidenum + 2]; j++) {
				cout << ba[hidenum + 2][j][n] << "\t";
				btout << ba[hidenum + 2][j][n] << "\t";
				double temp = ba[hidenum + 2][j][n];
				if (temp > 0.5) {
					if (y[j][i + n - 1] == 1)
						count++;
				}
			}
			cout << endl;
			btout << endl;
		}

	}
	double accuracy;
	accuracy = double(count) /double(tnum-trainnum);
	cout << "判断正确个数：" << count << "正确率：" << accuracy << endl;
	btout << "判断正确个数：" << count << "正确率：" << accuracy << endl;	
}
void normalall() {
	for (int i = 1; i <= tnum; i++) {
		double max = x[1][i];
		double min = x[1][i];
		for (int j = 2; j <= num[1]; j++) {//寻找最大值和最小值
			if (x[j][i] > max)
				max = x[j][i];
			else if (x[j][i] < min) {
				min = x[j][i];
			}
		}
		//cout << max << "\t" << min<<endl;
		for (int j = 1; j <= num[1]; j++) {
			nmx[j][i] = (x[j][i] - min + 1) / (max - min + 1);
			//cout << nmx[j][i] << "\t";
			//xx[i] = (xx[i] - min + 5) / (max - xx[i] + 5);
		}
		//cout << endl;
	}
	/*for (int i = 1; i <= tnum; i++) {
		for (int j = 1; j <= num[1]; j++) {
			cout << nmx[j][i]<<"\t";
		}
		cout << endl;
	}*/


}
bool dobatchtrain(int traintime, double acc, int& times) {
	double e = 0; int time;
	for (time = 1; time <= traintime; time++) {
		e = batchtrain();
		if (time <= 10000) {
			if (time % 500 == 0)
				printf("%d\t%0.12f\n", time, e);
		}
		else {
			if (time % 5000 == 0)
				printf("%d\t%0.12f\n", time, e);
		}//每500次或者5000次输出一次误差

		if (e <= acc) {
			break;
			printf("%d:\t%.12f\n", time, e);
		}
	}
	if (e <= acc) {//如果误差小于精度，迭代成功，模型收敛
		times = time;//输出迭代次数
		return true;
	}
	else {//到达最大迭代次数仍然发散，迭代失败
		times = time;
		return false;
	}
}
double batchtrain() {
	double e = 0;//误差
	//int r = rand() % 6 + 1;

	//r = r % 6;

	for (int i = r; i <= trainnum; i += batchsize) {
		//batchpositive(i);
		batchpos(i, batchsize);
		for (int l = 1; l <= batchsize; l++) {//误差计算
			for (int j = 1; j <= num[hidenum + 2]; j++) {
				double temp = ba[hidenum + 2][j][l] - y[j][i + l - 1];
				e = e + 1.0 / 2 * temp * temp;
			}
		}
		batchnega(i, batchsize);
		//batchnegative(i);
	}
	//r++;

	return e;
}
void batchpos(int begin, int batchsize) {
	for (int n = 1; n <= batchsize; n++) {
		for (int i = 1; i <= num[1]; i++) {//输入层
			bz[1][i][n] = nmx[i][n + begin - 1];
			ba[1][i][n] = bz[1][i][n];
		}
		for (int l = 2; l <= hidenum + 1; l++) {//后续隐藏层
			for (int i = 1; i <= num[l]; i++) {
				//bz[l][i][n] = b[l - 1];
				bz[l][i][n] = bb[l][i];
				for (int j = 1; j <= num[l - 1]; j++) {
					bz[l][i][n] += w[l - 1][i][j] * ba[l - 1][j][n];
				}//净输出
				ba[l][i][n] = f(bz[l][i][n], 3);//输出
			}
		}
		for (int i = 1; i <= num[hidenum + 2]; i++) {//输出层
			bz[hidenum + 2][i][n] = bb[hidenum + 2 ][i];
			for (int j = 1; j <= num[hidenum + 2 - 1]; j++) {
				bz[hidenum + 2][i][n] += w[hidenum + 2 - 1][i][j] * ba[hidenum + 2 - 1][j][n];
			}//净输出
			ba[hidenum + 2][i][n] = f(bz[hidenum + 2][i][n], 3);//输出
		}
	}

}
void batchnega(int begin, int batchsize) {
	for (int n = 1; n <= batchsize; n++) {
		//cout << n << endl;
		for (int i = 1; i <= num[hidenum + 2]; i++) {

			bd[hidenum + 2][i][n] = df(bz[hidenum + 2][i][n], 3) * (ba[hidenum + 2][i][n] - y[i][n + begin - 1]);
			//cout << bd[hidenum + 2][i][n] << "\t";
		}
		//cout << endl;
		for (int l = hidenum + 1; l >= 2; l--) {//隐藏层误差
			for (int i = 1; i <= num[l]; i++) {
				bd[l][i][n] = 0;
				for (int k = 1; k <= num[l + 1]; k++) {
					bd[l][i][n] += bd[l + 1][k][n] * w[l][k][i];
				}
				bd[l][i][n] *= df(bz[l][i][n], 3);
			}
		}

	}
	for (int l = hidenum + 2; l >= 2; l--) {//误差反传
		for (int i = 1; i <= num[l]; i++) {
			for (int j = 1; j < num[l - 1]; j++) {
				double temp = 0;
				for (int n = 1; n <= batchsize; n++) {
					temp = temp + bd[l][i][n] * ba[l - 1][j][n];
				}
				w[l - 1][i][j] -= tau * 1.0 / batchsize * temp;
			}
		}
	}
	double temp = 0.0;
	for (int l = hidenum + 2; l >= 2; l--) {//偏置更新
		//
		for (int i = 1; i <= num[l]; i++) {
			temp = 0.0;
			for (int n = 1; n <= batchsize; n++) {
				temp += bd[l][i][n];
			}
			bb[l][i] -= tau * temp / batchsize;
		}
		//b[l - 1] -= tau * temp / num[l]/batchsize;
	}

}
