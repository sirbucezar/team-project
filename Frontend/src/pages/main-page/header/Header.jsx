import React from 'react';
import s from './styles.module.scss';
import Avatar from '/avatar.jpg';

const Header = ({ isSidebarShow, setIsSidebarShow }) => {
   const handleBurgerClick = () => {
      setIsSidebarShow((prev) => !prev);
   };
   return (
      <div className={s.header}>
         <div className={s.header__container}>
            <div className={s.header__burger} onClick={handleBurgerClick}>
               <div className={`${s.burger} ${isSidebarShow ? s.active : ''}`}>
                  <span></span>
                  <span></span>
                  <span></span>
               </div>
            </div>
            <div className={s.header__icon}>
               <div className={s.header__img}>
                  <img src={Avatar} />
               </div>
               <ul className={s.header__menu}>
                  <li className={s.header__item}></li>
               </ul>
            </div>
         </div>
      </div>
   );
};

export default Header;
