import React from 'react';
import NavbarItem from '@theme-original/NavbarItem';
import AuthNavbar from '@site/src/components/Auth/AuthNavbar';

export default function NavbarItemWrapper(props) {
  if (props.type === 'custom-auth') {
    return <AuthNavbar {...props} />;
  }
  return <NavbarItem {...props} />;
}
