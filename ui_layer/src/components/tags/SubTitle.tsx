const SubTitle = ({subTitle,className}:{subTitle:string,className?:string}) => {
  return (
    <h1 className={`${className} text font-semibold text-xl capitalize`}>{subTitle}</h1>
  )
}

export default SubTitle