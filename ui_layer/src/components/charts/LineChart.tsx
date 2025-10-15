import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type LineChartProps = {
    data:{
        name:string,
        uv:number,
        pv?:number,
        amt?:number
    }[],
    graphColor:string
}



const LineChart = ({data,graphColor}:LineChartProps) => {

    const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
      if (active && payload && payload.length) {
      const value = payload[0].value;

      const label = payload[0].payload.name
      return (
        <div className="foreground rounded-lg py-2 px-3 text-sm ">
          <p className="paragraph text-xs">{label}</p>
          <p className="text-white font-medium">{`${value}`}</p>
        </div>
      );
    }
    return null;
  };


  return (
    <div className='chart-size'>
        <ResponsiveContainer width={'100%'} height={'100%'}>
            <AreaChart width={500} height={400} data={data} margin={{top:10,right:20,left:-20,bottom:0}}>
                <CartesianGrid strokeDasharray={'3 3'} vertical={false}/>
                <XAxis 
                dataKey={'name'}
                tickMargin={15} 
                tickLine={false}
                tick={{ fontSize: 12 }} />
                <YAxis
                axisLine={false}
                tickMargin={15}
                tickLine={false}
                tick={{ fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area type={'monotone'} dataKey={'uv'} stroke={graphColor} fill ={graphColor}/>
            </AreaChart>
        </ResponsiveContainer>
    </div>
  )
}

export default LineChart