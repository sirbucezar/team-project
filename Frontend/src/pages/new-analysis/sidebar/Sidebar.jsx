import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import s from './styles.module.scss';
import discursThrow from '/icons/discus throw.svg';
import highJump from '/icons/high jump.svg';
import hurdles from '/icons/hurdles 2.0.svg';
import javelinThrow from '/icons/javelin throw.svg';
import longJump from '/icons/long jump 2.0.svg';
import relayRace from '/icons/relay race 2.0.svg';
import shotPot from '/icons/shot pot 2.0.svg';
import sprint from '/icons/sprint.svg';
import startTecchnique from '/icons/start tecchnique.svg';
import { toast } from 'sonner';

const Sidebar = ({ rubrics, currentRubric, setCurrentRubric, isSidebarShow, setIsSidebarShow }) => {
   const iconsSvg = [
      startTecchnique,
      sprint,
      shotPot,
      highJump,
      hurdles,
      longJump,
      discursThrow,
      javelinThrow,
      relayRace,
   ];

   const sidebarRef = useRef(null);
   const sidebarBgRef = useRef(null);
   const dockRef = useRef(null);
   const iconRefs = useRef([]);
   const [isHeightExceeded, setIsHeightExceeded] = useState(false);
   const [sidebarWidth, setSidebarWidth] = useState(0);

   useEffect(() => {
      const icons = iconRefs.current;
      const dock = dockRef.current;
      const min = 80; // width + margin
      const max = 120;
      const bound = min * Math.PI;

      // Check if sidebar height exceeds screen height
      const checkHeight = () => {
         if (dock.scrollHeight > window.innerHeight || window.innerWidth < 992) {
            setIsHeightExceeded(true);
         } else {
            setIsHeightExceeded(false);
         }
      };

      // Initial check and add resize listener
      checkHeight();
      window.addEventListener('resize', checkHeight);

      gsap.set(icons, {
         transformOrigin: '0% -0%',
         height: 60,
      });

      gsap.set(dock, {
         position: 'relative',
      });

      const updateIcons = (pointer) => {
         icons.forEach((icon, i) => {
            let distance = i * min + min / 2 - pointer;
            let y = 0;
            let scale = 1;

            if (Math.abs(distance) < bound) {
               let rad = (distance / min) * 0.5;
               scale = 1 + (max / min - 1) * Math.cos(rad);
               y = (max - min) * Math.sin(rad);
               icon.classList.add(s.show); // Add active class to hovered item
               icon.classList.remove(s.inactive);
            } else {
               icon.classList.add(s.inactive); // Add inactive class to other items
               icon.classList.remove(s.show);
            }

            gsap.to(icon, {
               duration: 0.3,
               y: y,
               scale: scale,
            });
         });
      };

      const handleMouseMove = (event) => {
         if (!isHeightExceeded) {
            const offset = dock.getBoundingClientRect().top;
            const mouseY = event.clientY - offset;
            updateIcons(mouseY);
         }
      };

      const handleMouseLeave = () => {
         if (!isHeightExceeded) {
            gsap.to(icons, {
               duration: 0.3,
               scale: 1,
               y: 0,
            });

            // Remove all active and inactive classes
            icons.forEach((icon) => {
               icon.classList.remove(s.show, s.inactive);
            });
         }
      };

      dock.addEventListener('mousemove', handleMouseMove);
      dock.addEventListener('mouseleave', handleMouseLeave);

      return () => {
         if (!isHeightExceeded) {
            dock.removeEventListener('mousemove', handleMouseMove);
            dock.removeEventListener('mouseleave', handleMouseLeave);
         }
         window.removeEventListener('resize', checkHeight);
      };
   }, [isHeightExceeded]);

   useEffect(() => {
      const updateSidebarWidth = () => {
         if (sidebarRef.current && sidebarBgRef.current) {
            const sidebarW = sidebarRef.current.offsetWidth;
            setSidebarWidth(sidebarW);
            sidebarBgRef.current.style.width = `calc(100% - ${sidebarW}px)`;
         }
      };

      // Initial set
      updateSidebarWidth();

      // Update width on resize
      window.addEventListener('resize', updateSidebarWidth);

      return () => {
         window.removeEventListener('resize', updateSidebarWidth);
      };
   }, []);

   const handleRubricClick = (rubric) => {
      setCurrentRubric(rubric);
      toast.success(`Rubric ${rubric.name}`);
      setIsSidebarShow(false);
   };

   const closeSidebar = () => {
      setIsSidebarShow(false);
   };

   return (
      <>
         <div
            className={`${s.sidebarBg} ${isSidebarShow ? s.active : ''}`}
            ref={sidebarBgRef}
            onClick={closeSidebar}></div>
         <div className={`${s.sidebar} ${isSidebarShow ? s.show : ''}`} ref={sidebarRef}>
            <div className={`${s.sidebar__wrapper} ${isHeightExceeded ? s.scroll : ''}`}>
               <ul className={s.sidebar__toolbar} ref={dockRef}>
                  {iconsSvg.map((icon, index) => (
                     <li
                        key={index}
                        onClick={() =>
                           handleRubricClick({ id: rubrics[index].id, name: rubrics[index].name })
                        }
                        className={`${s.sidebar__item} ${
                           iconRefs.current[index]?.classList.contains(s.show) ? s.show : ''
                        } ${
                           iconRefs.current[index]?.classList.contains(s.inactive) ? s.inactive : ''
                        } ${currentRubric?.id === rubrics[index].id ? s.active : ''}`}
                        ref={(el) => (iconRefs.current[index] = el)}>
                        <div className={s.sidebar__icon}>
                           <img src={icon} alt={`icon-${index}`} />
                        </div>
                        <div className={s.sidebar__title}>{rubrics[index].name}</div>
                     </li>
                  ))}
               </ul>
            </div>
         </div>
      </>
   );
};

export default Sidebar;
