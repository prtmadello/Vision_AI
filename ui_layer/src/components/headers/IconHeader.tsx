import type { IconType } from 'react-icons'
import SubTitle from '../tags/SubTitle'

const IconHeader = ({headerData,className}:{headerData:{title:string,paragraph?:string,icon:IconType,iconStyle:string,titleStyle?:string,paragraphStyle?:string},className?:string}) => {
  return (
    <div className={`${className} gap-3 flex  `}>
        <span className={` ${headerData.iconStyle} p-2 text rounded-lg flex justify-center items-center text-2xl`}><headerData.icon className=''/></span>
        <div className="">
          <SubTitle subTitle={headerData.title} className={`${headerData.titleStyle} !text-lg`}/>
          <p className={`paragraph !text-xs ${headerData.paragraphStyle}`}>{headerData.paragraph}</p>
        </div>
    </div>
  )
}

export default IconHeader