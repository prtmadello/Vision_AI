import { PieChart as PieChartRe, ResponsiveContainer, Tooltip, Pie, Cell } from 'recharts'

type PieChartProps = {
  data: {
    name: string
    value: number
    color: string
  }[],
  graphColor?: string
}

const PieChart = ({ data }: PieChartProps) => {

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
    <div className="w-full h-[78%] xl:pb-3 xxl:pb-0">
      <ResponsiveContainer width="100%" height="100%">
        <PieChartRe>
          <Tooltip content={<CustomTooltip />} />
          <Pie
            data={data}
            dataKey="value"     
            nameKey="name"
            cx="50%"
            cy="50%"
            fill="#01B075"
            label={false}
            labelLine={false}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
        </PieChartRe>
      </ResponsiveContainer>
    </div>
  )
}

export default PieChart
